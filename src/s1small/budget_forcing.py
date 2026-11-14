"""Backend-independent, bounded s1 reasoning/answer state machine."""

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Segment:
    tokens: tuple
    timed_out: bool = False


@dataclass(frozen=True)
class Protocol:
    reasoning_end: int
    eos_ids: tuple
    answer_header: tuple
    newline: tuple
    wait: tuple


@dataclass(frozen=True)
class Budget:
    condition: str
    reasoning: int
    answer: int
    absolute: int
    waits: int = 0


CONDITIONS = ("A", "B", "C", "D256", "D512", "D1024", "D2048", "E1", "E2")


def condition_budget(condition, natural_limit, answer_limit, absolute_limit):
    if condition not in CONDITIONS:
        raise ValueError("Unknown experiment condition")
    if condition in ("A", "B"):
        return None
    reasoning = int(condition[1:]) if condition.startswith("D") else natural_limit
    waits = int(condition[1:]) if condition.startswith("E") else 0
    return Budget(condition, reasoning, answer_limit, absolute_limit, waits)


def run_structured(decoder, prompt_tokens, budget, protocol, timeout_seconds):
    """Generate reasoning, intercept a real stop, optionally Wait, then read out.

    Decoder exposes generate(context, maximum, stop_ids, deadline) -> Segment and
    decode(tokens) -> visible text. Returned tokens include any emitted stop.
    Every emitted token counts, including markers subsequently removed from the
    context. Control injections count separately and against the absolute limit.
    """
    reserve = budget.answer + len(protocol.newline) + len(protocol.answer_header)
    if (budget.reasoning < 0 or budget.answer < 1 or budget.waits < 0
            or budget.absolute < reserve or timeout_seconds <= 0):
        raise ValueError("Invalid budgets or insufficient room reserved for the answer")
    started = time.perf_counter()
    deadline = started + timeout_seconds
    context = list(prompt_tokens)
    events = []
    generated = injected = reason_phase = reason_content = answer_tokens = waits = 0
    budget_hit = safety_hit = timed_out = early_eos = False
    stop_reason = None
    wait_denied = False

    def call(phase, maximum, stop_ids):
        nonlocal generated, timed_out
        segment = decoder.generate(context.copy(), maximum, set(stop_ids), deadline)
        tokens = list(segment.tokens)
        if (len(tokens) > maximum or any(not isinstance(t, int) or t < 0 for t in tokens)
                or any(t in stop_ids for t in tokens[:-1])):
            raise ValueError("Decoder violated the bounded stop-token contract")
        if not tokens and not segment.timed_out:
            raise RuntimeError("Decoder returned an unexplained empty segment")
        generated += len(tokens)
        timed_out = timed_out or segment.timed_out
        events.append({"origin": "model", "phase": phase, "token_ids": tokens,
                       "text": decoder.decode(tokens), "timed_out": segment.timed_out})
        context.extend(tokens)
        return tokens

    def inject(phase, tokens):
        nonlocal injected
        if generated + injected + len(tokens) > budget.absolute:
            raise ValueError("Control injection would exceed absolute safety ceiling")
        context.extend(tokens)
        injected += len(tokens)
        events.append({"origin": "injected", "phase": phase, "token_ids": list(tokens),
                       "text": decoder.decode(tokens)})

    stops = set(protocol.eos_ids) | {protocol.reasoning_end}
    while True:
        remaining_reason = budget.reasoning - reason_phase
        available = budget.absolute - generated - injected - reserve
        maximum = min(remaining_reason, max(0, available))
        if maximum <= 0:
            budget_hit = remaining_reason <= 0
            safety_hit = available <= 0 and not budget_hit
            stop_reason = "reasoning_budget" if budget_hit else "absolute_answer_reservation"
            break
        tokens = call("reasoning", maximum, stops)
        reason_phase += len(tokens)
        stopped = bool(tokens) and tokens[-1] in stops
        reason_content += len(tokens) - int(stopped)
        if stopped:
            # Preserve the emitted stop in events/accounting, but remove it from
            # the next input before injecting a continuation or common readout.
            marker = context.pop()
            early_eos = early_eos or marker in protocol.eos_ids
            events[-1]["removed_stop_token_id"] = marker
            remaining_reason = budget.reasoning - reason_phase
            enough = budget.absolute - generated - injected - len(protocol.wait) - reserve >= 1
            if waits < budget.waits and remaining_reason > 0 and enough and not timed_out:
                inject("wait", protocol.wait)
                waits += 1
                continue
            wait_denied = waits < budget.waits and (remaining_reason <= 0 or not enough)
            stop_reason = "eos" if marker in protocol.eos_ids else "reasoning_marker"
            break
        if timed_out:
            stop_reason = "timeout"
            break
        if len(tokens) != maximum:
            raise RuntimeError("Decoder ended reasoning without a stop, cap or deadline")
        budget_hit = reason_phase >= budget.reasoning
        safety_hit = not budget_hit
        stop_reason = "reasoning_budget" if budget_hit else "absolute_answer_reservation"
        break

    body = []
    answer_stop = None
    if not timed_out:
        prefix = () if decoder.decode(context).endswith("\n") else protocol.newline
        inject("answer_transition", (*prefix, *protocol.answer_header))
        maximum = min(budget.answer, budget.absolute - generated - injected)
        if maximum > 0:
            body = call("answer", maximum, stops)
            answer_tokens = len(body)
            if body and body[-1] in stops:
                answer_stop = "eos" if body[-1] in protocol.eos_ids else "unexpected_role_marker"
            elif timed_out:
                answer_stop = "timeout"
            else:
                answer_stop = "answer_budget"
        else:
            answer_stop = "absolute_safety"
            safety_hit = True
    else:
        answer_stop = "timeout_before_answer"
    clean_body = body[:-1] if body and body[-1] in stops else body
    if generated + injected > budget.absolute:
        raise AssertionError("Absolute safety invariant failed")
    return {"condition": budget.condition, "status": "ok", "segments": events,
            "answer_text": decoder.decode(clean_body), "answer_token_ids": body,
            "reasoning_tokens": reason_content, "reasoning_phase_generated_tokens": reason_phase,
            "answer_generated_tokens": answer_tokens, "total_generated_tokens": generated,
            "injected_tokens": injected, "total_new_tokens_consumed": generated + injected,
            "forced_wait_count": waits, "wait_denied_exhausted_budget": wait_denied,
            "reasoning_budget": budget.reasoning, "answer_budget": budget.answer,
            "absolute_safety_ceiling": budget.absolute, "reasoning_termination": stop_reason,
            "termination_reason": answer_stop, "reasoning_cap_hit": budget_hit,
            "safety_cap_hit": safety_hit or (budget_hit and not budget.condition.startswith("D")),
            "answer_cap_hit": answer_stop == "answer_budget", "early_reasoning_eos": early_eos,
            "timed_out": timed_out, "latency_seconds": time.perf_counter() - started,
            "final_context_token_ids": context,
            "visible_transcript": decoder.decode(context[len(prompt_tokens):])}

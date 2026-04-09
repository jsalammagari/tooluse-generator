"""Conversation state machine.

:class:`ConversationStateMachine` formalizes the orchestrator's control
flow into an explicit set of :class:`ConversationState` values and
:class:`ConversationEvent` triggers, with a validated transition table.

The state machine is *additive* — it tracks and validates transitions
alongside the existing orchestrator logic without replacing it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConversationState(str, Enum):
    """States in a conversation."""

    INIT = "init"
    USER_TURN = "user_turn"
    ASSISTANT_TURN = "assistant_turn"
    TOOL_EXECUTION = "tool_execution"
    DISAMBIGUATION = "disambiguation"
    COMPLETE = "complete"
    FAILED = "failed"


class ConversationEvent(str, Enum):
    """Events that trigger state transitions."""

    START = "start"
    USER_MESSAGE = "user_message"
    ASSISTANT_TEXT = "assistant_text"
    ASSISTANT_TOOL_CALL = "assistant_tool_call"
    ASSISTANT_DISAMBIGUATE = "assistant_disambiguate"
    ASSISTANT_FINAL = "assistant_final"
    TOOL_RESULT = "tool_result"
    USER_CLARIFICATION = "user_clarification"
    MAX_TURNS_REACHED = "max_turns_reached"
    CHAIN_COMPLETE = "chain_complete"
    ERROR = "error"


# ---------------------------------------------------------------------------
# StateTransition model
# ---------------------------------------------------------------------------


class StateTransition(BaseModel):
    """A single allowed transition."""

    model_config = ConfigDict(use_enum_values=True)

    from_state: ConversationState = Field(..., description="Source state.")
    to_state: ConversationState = Field(..., description="Target state.")
    event: ConversationEvent = Field(..., description="Triggering event.")
    description: str = Field(default="", description="Human-readable note.")


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class InvalidTransitionError(Exception):
    """Raised when a transition is not allowed."""


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

_TRANSITIONS: list[tuple[ConversationState, ConversationEvent, ConversationState]] = [
    # INIT
    (ConversationState.INIT, ConversationEvent.START, ConversationState.USER_TURN),
    # USER_TURN
    (ConversationState.USER_TURN, ConversationEvent.USER_MESSAGE, ConversationState.ASSISTANT_TURN),
    (ConversationState.USER_TURN, ConversationEvent.MAX_TURNS_REACHED, ConversationState.COMPLETE),
    (ConversationState.USER_TURN, ConversationEvent.CHAIN_COMPLETE, ConversationState.COMPLETE),
    (ConversationState.USER_TURN, ConversationEvent.ERROR, ConversationState.FAILED),
    # ASSISTANT_TURN
    (ConversationState.ASSISTANT_TURN, ConversationEvent.ASSISTANT_TOOL_CALL, ConversationState.TOOL_EXECUTION),
    (ConversationState.ASSISTANT_TURN, ConversationEvent.ASSISTANT_DISAMBIGUATE, ConversationState.DISAMBIGUATION),
    (ConversationState.ASSISTANT_TURN, ConversationEvent.ASSISTANT_FINAL, ConversationState.COMPLETE),
    (ConversationState.ASSISTANT_TURN, ConversationEvent.ASSISTANT_TEXT, ConversationState.USER_TURN),
    (ConversationState.ASSISTANT_TURN, ConversationEvent.MAX_TURNS_REACHED, ConversationState.COMPLETE),
    (ConversationState.ASSISTANT_TURN, ConversationEvent.CHAIN_COMPLETE, ConversationState.COMPLETE),
    (ConversationState.ASSISTANT_TURN, ConversationEvent.ERROR, ConversationState.FAILED),
    # TOOL_EXECUTION
    (ConversationState.TOOL_EXECUTION, ConversationEvent.TOOL_RESULT, ConversationState.USER_TURN),
    (ConversationState.TOOL_EXECUTION, ConversationEvent.CHAIN_COMPLETE, ConversationState.COMPLETE),
    (ConversationState.TOOL_EXECUTION, ConversationEvent.ERROR, ConversationState.FAILED),
    # DISAMBIGUATION
    (ConversationState.DISAMBIGUATION, ConversationEvent.USER_CLARIFICATION, ConversationState.ASSISTANT_TURN),
    (ConversationState.DISAMBIGUATION, ConversationEvent.MAX_TURNS_REACHED, ConversationState.COMPLETE),
    (ConversationState.DISAMBIGUATION, ConversationEvent.ERROR, ConversationState.FAILED),
]


# ---------------------------------------------------------------------------
# ConversationStateMachine
# ---------------------------------------------------------------------------


class ConversationStateMachine:
    """Manages conversation state transitions."""

    def __init__(self) -> None:
        self._state = ConversationState.INIT
        self._history: list[
            tuple[ConversationState, ConversationEvent, ConversationState]
        ] = []
        self._transitions: dict[
            tuple[ConversationState, ConversationEvent], ConversationState
        ] = {(fr, ev): to for fr, ev, to in _TRANSITIONS}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConversationState:
        """Current state."""
        return self._state

    @property
    def history(
        self,
    ) -> list[tuple[ConversationState, ConversationEvent, ConversationState]]:
        """Copy of the transition history."""
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """True when in COMPLETE or FAILED."""
        return self._state in (ConversationState.COMPLETE, ConversationState.FAILED)

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(self, event: ConversationEvent) -> ConversationState:
        """Apply *event* and return the new state.

        Raises :class:`InvalidTransitionError` if the transition is not
        allowed from the current state.
        """
        key = (self._state, event)
        new_state = self._transitions.get(key)
        if new_state is None:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.value} via {event.value}"
            )
        old = self._state
        self._state = new_state
        self._history.append((old, event, new_state))
        return new_state

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_valid_events(self) -> list[ConversationEvent]:
        """Events valid from the current state."""
        return [ev for (st, ev) in self._transitions if st == self._state]

    def get_valid_transitions(self) -> list[ConversationState]:
        """States reachable from the current state."""
        return [to for (st, _), to in self._transitions.items() if st == self._state]

    def can_transition(self, event: ConversationEvent) -> bool:
        """Return ``True`` if *event* is valid from the current state."""
        return (self._state, event) in self._transitions

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to INIT and clear history."""
        self._state = ConversationState.INIT
        self._history.clear()

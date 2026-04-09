"""Tests for the ConversationStateMachine (Task 40)."""

from __future__ import annotations

import pytest

from tooluse_gen.agents.state_machine import (
    ConversationEvent,
    ConversationState,
    ConversationStateMachine,
    InvalidTransitionError,
    StateTransition,
)

pytestmark = pytest.mark.unit


# ===================================================================
# ConversationState enum
# ===================================================================


class TestConversationState:
    def test_all_values_exist(self):
        assert ConversationState.INIT == "init"
        assert ConversationState.USER_TURN == "user_turn"
        assert ConversationState.ASSISTANT_TURN == "assistant_turn"
        assert ConversationState.TOOL_EXECUTION == "tool_execution"
        assert ConversationState.DISAMBIGUATION == "disambiguation"
        assert ConversationState.COMPLETE == "complete"
        assert ConversationState.FAILED == "failed"

    def test_values_are_strings(self):
        for s in ConversationState:
            assert isinstance(s.value, str)

    def test_terminal_states(self):
        terminals = {ConversationState.COMPLETE, ConversationState.FAILED}
        non_terminals = set(ConversationState) - terminals
        assert len(terminals) == 2
        assert len(non_terminals) == 5


# ===================================================================
# ConversationEvent enum
# ===================================================================


class TestConversationEvent:
    def test_all_values_exist(self):
        assert len(ConversationEvent) == 11
        assert ConversationEvent.START == "start"
        assert ConversationEvent.USER_MESSAGE == "user_message"
        assert ConversationEvent.ASSISTANT_TEXT == "assistant_text"
        assert ConversationEvent.ASSISTANT_TOOL_CALL == "assistant_tool_call"
        assert ConversationEvent.ASSISTANT_DISAMBIGUATE == "assistant_disambiguate"
        assert ConversationEvent.ASSISTANT_FINAL == "assistant_final"
        assert ConversationEvent.TOOL_RESULT == "tool_result"
        assert ConversationEvent.USER_CLARIFICATION == "user_clarification"
        assert ConversationEvent.MAX_TURNS_REACHED == "max_turns_reached"
        assert ConversationEvent.CHAIN_COMPLETE == "chain_complete"
        assert ConversationEvent.ERROR == "error"

    def test_values_are_strings(self):
        for e in ConversationEvent:
            assert isinstance(e.value, str)


# ===================================================================
# StateTransition model
# ===================================================================


class TestStateTransition:
    def test_construction(self):
        t = StateTransition(
            from_state=ConversationState.INIT,
            to_state=ConversationState.USER_TURN,
            event=ConversationEvent.START,
            description="Conversation begins",
        )
        assert t.from_state == "init"
        assert t.to_state == "user_turn"
        assert t.event == "start"
        assert t.description == "Conversation begins"

    def test_default_description(self):
        t = StateTransition(
            from_state=ConversationState.INIT,
            to_state=ConversationState.USER_TURN,
            event=ConversationEvent.START,
        )
        assert t.description == ""


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_initial_state(self):
        sm = ConversationStateMachine()
        assert sm.state == ConversationState.INIT

    def test_empty_history(self):
        sm = ConversationStateMachine()
        assert sm.history == []

    def test_not_terminal(self):
        sm = ConversationStateMachine()
        assert sm.is_terminal is False


# ===================================================================
# Transitions — happy path
# ===================================================================


class TestTransitionHappy:
    def test_init_to_user_turn(self):
        sm = ConversationStateMachine()
        result = sm.transition(ConversationEvent.START)
        assert result == ConversationState.USER_TURN
        assert sm.state == ConversationState.USER_TURN

    def test_user_turn_to_assistant(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        result = sm.transition(ConversationEvent.USER_MESSAGE)
        assert result == ConversationState.ASSISTANT_TURN

    def test_assistant_to_tool_execution(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        result = sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        assert result == ConversationState.TOOL_EXECUTION

    def test_assistant_to_disambiguation(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        result = sm.transition(ConversationEvent.ASSISTANT_DISAMBIGUATE)
        assert result == ConversationState.DISAMBIGUATION

    def test_assistant_to_complete(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        result = sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert result == ConversationState.COMPLETE
        assert sm.is_terminal

    def test_assistant_text_to_user_turn(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        result = sm.transition(ConversationEvent.ASSISTANT_TEXT)
        assert result == ConversationState.USER_TURN

    def test_tool_result_to_user_turn(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        result = sm.transition(ConversationEvent.TOOL_RESULT)
        assert result == ConversationState.USER_TURN

    def test_tool_chain_complete(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        result = sm.transition(ConversationEvent.CHAIN_COMPLETE)
        assert result == ConversationState.COMPLETE

    def test_disambiguation_to_assistant(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_DISAMBIGUATE)
        result = sm.transition(ConversationEvent.USER_CLARIFICATION)
        assert result == ConversationState.ASSISTANT_TURN

    def test_full_conversation_path(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        sm.transition(ConversationEvent.TOOL_RESULT)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert sm.state == ConversationState.COMPLETE
        assert sm.is_terminal
        assert len(sm.history) == 6


# ===================================================================
# Transitions — invalid
# ===================================================================


class TestTransitionInvalid:
    def test_init_user_message(self):
        sm = ConversationStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition(ConversationEvent.USER_MESSAGE)

    def test_complete_any_event(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert sm.is_terminal
        with pytest.raises(InvalidTransitionError):
            sm.transition(ConversationEvent.USER_MESSAGE)

    def test_failed_any_event(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ERROR)
        assert sm.state == ConversationState.FAILED
        with pytest.raises(InvalidTransitionError):
            sm.transition(ConversationEvent.START)

    def test_user_turn_tool_result(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        with pytest.raises(InvalidTransitionError):
            sm.transition(ConversationEvent.TOOL_RESULT)


# ===================================================================
# is_terminal
# ===================================================================


class TestIsTerminal:
    def test_non_terminal_states(self):
        sm = ConversationStateMachine()
        assert sm.is_terminal is False  # INIT
        sm.transition(ConversationEvent.START)
        assert sm.is_terminal is False  # USER_TURN
        sm.transition(ConversationEvent.USER_MESSAGE)
        assert sm.is_terminal is False  # ASSISTANT_TURN
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        assert sm.is_terminal is False  # TOOL_EXECUTION

    def test_complete_is_terminal(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert sm.is_terminal is True

    def test_failed_is_terminal(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ERROR)
        assert sm.is_terminal is True


# ===================================================================
# get_valid_events
# ===================================================================


class TestGetValidEvents:
    def test_from_init(self):
        sm = ConversationStateMachine()
        events = sm.get_valid_events()
        assert events == [ConversationEvent.START]

    def test_from_assistant_turn(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        events = sm.get_valid_events()
        assert ConversationEvent.ASSISTANT_TOOL_CALL in events
        assert ConversationEvent.ASSISTANT_DISAMBIGUATE in events
        assert ConversationEvent.ASSISTANT_FINAL in events
        assert ConversationEvent.ASSISTANT_TEXT in events
        assert ConversationEvent.MAX_TURNS_REACHED in events
        assert ConversationEvent.CHAIN_COMPLETE in events
        assert ConversationEvent.ERROR in events

    def test_from_complete(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert sm.get_valid_events() == []

    def test_from_tool_execution(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        events = sm.get_valid_events()
        assert ConversationEvent.TOOL_RESULT in events
        assert ConversationEvent.CHAIN_COMPLETE in events
        assert ConversationEvent.ERROR in events


# ===================================================================
# get_valid_transitions
# ===================================================================


class TestGetValidTransitions:
    def test_from_init(self):
        sm = ConversationStateMachine()
        targets = sm.get_valid_transitions()
        assert targets == [ConversationState.USER_TURN]

    def test_from_assistant_turn(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        targets = sm.get_valid_transitions()
        assert ConversationState.TOOL_EXECUTION in targets
        assert ConversationState.DISAMBIGUATION in targets
        assert ConversationState.COMPLETE in targets
        assert ConversationState.USER_TURN in targets
        assert ConversationState.FAILED in targets


# ===================================================================
# can_transition
# ===================================================================


class TestCanTransition:
    def test_valid(self):
        sm = ConversationStateMachine()
        assert sm.can_transition(ConversationEvent.START) is True

    def test_invalid(self):
        sm = ConversationStateMachine()
        assert sm.can_transition(ConversationEvent.USER_MESSAGE) is False

    def test_terminal(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert sm.can_transition(ConversationEvent.START) is False
        assert sm.can_transition(ConversationEvent.USER_MESSAGE) is False


# ===================================================================
# History
# ===================================================================


class TestHistory:
    def test_empty_initially(self):
        sm = ConversationStateMachine()
        assert sm.history == []

    def test_records_transitions(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        assert len(sm.history) == 1
        old, ev, new = sm.history[0]
        assert old == ConversationState.INIT
        assert ev == ConversationEvent.START
        assert new == ConversationState.USER_TURN

    def test_full_path_recorded(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_DISAMBIGUATE)
        sm.transition(ConversationEvent.USER_CLARIFICATION)
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        sm.transition(ConversationEvent.TOOL_RESULT)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert len(sm.history) == 8
        assert sm.history[0][0] == ConversationState.INIT
        assert sm.history[-1][2] == ConversationState.COMPLETE


# ===================================================================
# Reset
# ===================================================================


class TestReset:
    def test_resets_to_init(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.reset()
        assert sm.state == ConversationState.INIT

    def test_clears_history(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.reset()
        assert sm.history == []

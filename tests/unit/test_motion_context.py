from dataclasses import replace

from agents.mechanism_state_tracker import GridLayout, MechanismObject, MechanismState
from agents.motion_context import MotionContextMemory


def small_object(cell=(1, 1), *, signature="tiny", object_id="moving", **kwargs):
    x, y = 5 * cell[0] + 1, 5 * cell[1] + 1
    return MechanismObject(
        object_id=object_id,
        cell=cell,
        color=2,
        area=3,
        shape_signature=signature,
        bbox=(x, y, x + 1, y + 1),
        **kwargs,
    )


def state(*objects, **kwargs):
    values = dict(
        level=4,
        game_state="NOT_FINISHED",
        layout=GridLayout((0, 0), (5, 5), 10, 10, 3, 4, 0.95),
        player_position=(8, 8),
        player_confidence=0.99,
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        objects=objects,
        resource_level=40,
        resource_capacity=84,
        status_pattern=((1,),),
        matching_target_ids=(),
    )
    values.update(kwargs)
    return MechanismState(**values)


def test_small_shuttle_same_current_position_has_different_motion_context():
    memory = MotionContextMemory()
    assert not memory.observe(state(small_object((1, 1))))
    right = memory.observe(state(small_object((2, 1))))
    memory.observe(state(small_object((3, 1))))
    left = memory.observe(state(small_object((2, 1))))
    assert right[0].current == left[0].current == (2, 1)
    assert right[0].previous == (1, 1)
    assert left[0].previous == (3, 1)
    assert right != left


def test_duplicate_component_identity_abstains_instead_of_matching_last():
    memory = MotionContextMemory()
    memory.observe(state(small_object((1, 1))))
    memory.observe(state(small_object((2, 1))))
    context = memory.observe(
        state(small_object((1, 1)), small_object((3, 1), object_id="copy"))
    )
    assert context[0].current is None
    resumed = memory.observe(state(small_object((2, 1))))
    assert resumed[0].previous is None


def test_occlusion_is_unknown_not_stationary_or_extrapolated():
    memory = MotionContextMemory()
    memory.observe(state(small_object((1, 1))))
    memory.observe(state(small_object((2, 1))))
    hidden = memory.observe(state())
    assert hidden[0].current is None
    assert hidden[0].previous == (2, 1)
    restored = memory.observe(state(small_object((3, 1))))
    assert restored[0].previous is None
    assert restored[0].current == (3, 1)


def test_stationary_components_do_not_expand_context():
    memory = MotionContextMemory()
    for _ in range(3):
        assert not memory.observe(state(small_object()))


def test_reset_level_change_and_empty_meter_clear_motion_history():
    for boundary in ("reset", "level", "resource"):
        memory = MotionContextMemory()
        memory.observe(state(small_object((1, 1))))
        memory.observe(
            state(
                small_object((2, 1)), resource_level=0 if boundary == "resource" else 40
            )
        )
        assert memory.context
        after = state(small_object((3, 1)), level=5 if boundary == "level" else 4)
        assert not memory.observe(after, reset=boundary == "reset")


def test_player_target_hud_and_oversized_components_are_not_tracks():
    components = (
        small_object((8, 8)),
        small_object((1, 1), signature="target", static_target=True),
        replace(small_object((2, 2)), object_id="player-band"),
        replace(small_object((3, 3)), bbox=(1, 60, 3, 62)),
        replace(small_object((4, 4)), bbox=(20, 20, 29, 29)),
    )
    observed = state(*components, player_component_ids=("player-band",))
    assert not MotionContextMemory.visible_positions(observed)

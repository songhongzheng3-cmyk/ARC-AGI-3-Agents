"""Regression coverage for touching sprites and evidence-scoped chains."""

from types import SimpleNamespace

import pytest

from agents.selectable_navigation_model import MultiActorState
from agents.shaped_contact_model import ShapedContactModel, ShapedSceneBinding


def picture(positions=((1, 4), (4, 4)), selected=0, cursor=False):
    grid = [[1] * 30 for _ in range(31)]
    for (cx, cy), (w, h) in zip(((1, 1), (7, 6)), ((3, 3), (6, 6))):
        x, y = cx * 3, cy * 3
        for dy in range(-1, h + 1):
            for dx in range(-1, w + 1):
                if dy in (-1, h) or dx in (-1, w):
                    grid[y + dy][x + dx] = 4
    for i, ((cx, cy), size) in enumerate(zip(positions, (3, 6))):
        x, y = cx * 3, cy * 3
        for dy in range(size):
            for dx in range(size):
                grid[y + dy][x + dx] = 14
        marker_size = size // 3
        offset = (size - marker_size) // 2
        for dy in range(marker_size):
            for dx in range(marker_size):
                grid[y + offset + dy][x + offset + dx] = (
                    0 if i == selected else 4 if size == 3 else 5
                )
        if cursor and i == selected:
            for dy in range(size):
                grid[y + dy][x] = 0
    return SimpleNamespace(frame=[grid])


def test_touching_colour_components_still_bind_two_shapes_and_selection():
    binding = ShapedSceneBinding(picture())
    assert binding.goals == ((1, 1), (7, 6))
    assert binding.parse(picture(((1, 4), (2, 4)), selected=1, cursor=True)) == (
        MultiActorState(((1, 4), (2, 4)), 1)
    )


def test_sprite_pattern_change_is_reported_instead_of_silent_identity_reuse():
    binding = ShapedSceneBinding(picture())
    changed = picture()
    changed.frame[-1][12][3] = 7
    with pytest.raises(ValueError, match="missing sprite"):
        binding.parse(changed)


def test_extra_unmatched_goal_is_not_silently_ignored():
    changed = picture()
    for dy in range(5):
        for dx in range(8):
            if dy in (0, 4) or dx in (0, 7):
                changed.frame[-1][dy + 1][dx + 16] = 4
    with pytest.raises(ValueError, match="unmatched outlines"):
        ShapedSceneBinding(changed)


def chain_model():
    dimensions = ((1, 1), (1, 2), (2, 1), (2, 2))
    binding = SimpleNamespace(
        floor_color=1,
        terrain={(x, y): 1 for x in range(20) for y in range(20)},
        goals=((8, 8), (2, 4), (10, 1), (4, 5)),
        footprint=lambda i, p: frozenset(
            (p[0] + dx, p[1] + dy)
            for dx in range(dimensions[i][0])
            for dy in range(dimensions[i][1])
        ),
    )
    model = ShapedContactModel(binding, {2: (-1, 0)})
    before = MultiActorState(((9, 5), (5, 4), (10, 1), (7, 5)), 0)
    after = MultiActorState(((9, 5), (2, 4), (10, 1), (4, 5)), 0)
    model.learn(before, 2, after)
    return model, before, after


def test_contact_can_move_two_passive_shapes_without_moving_selected_actor():
    model, before, after = chain_model()
    assert model.contact_destination(before, 2) == after
    assert model.ordinary_destination(before, 2) is None
    assert not any(t == "move:3" for t, _, _ in model.successors(before))
    assert ("move:2", after, 1) in list(model.successors(before))


@pytest.mark.parametrize("violation", ["terrain", "unobserved_chain", "relative_pose"])
def test_observed_chain_is_rejected_when_context_changes(violation):
    model, before, _ = chain_model()
    if violation == "terrain":
        model.binding.terrain[(3, 4)] = 15
    elif violation == "unobserved_chain":
        before = MultiActorState(((9, 5), (5, 4), (3, 5), (7, 5)), 0)
    else:
        before = MultiActorState(((9, 5), (5, 3), (10, 1), (7, 5)), 0)
    assert model.contact_destination(before, 2) is None


def test_conflicting_empirical_effects_block_execution():
    model, before, _ = chain_model()
    alternative = MultiActorState(((9, 5), (1, 4), (10, 1), (3, 5)), 0)
    model.learn(before, 2, alternative)
    with pytest.raises(ValueError, match="conflicting contact"):
        model.contact_destination(before, 2)


def test_navigation_macro_cost_counts_selection_and_every_primitive():
    model, _, _ = chain_model()
    model.directions = {2: (0, 1)}
    model.rules.clear()
    state = MultiActorState(((8, 7), (2, 4), (10, 1), (4, 5)), 1)
    macros = list(model.macro_successors(state))
    token, destination, cost = next(
        (t, s, c) for t, s, c in macros if s.positions[0] == (8, 8)
    )
    assert token == "select:0|move:2"
    assert cost == 2
    current = state
    for primitive in token.split("|"):
        current = {t: s for t, s, _ in model.successors(current)}[primitive]
    assert current == destination

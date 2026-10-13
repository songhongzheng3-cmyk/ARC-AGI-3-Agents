"""Check the evidence boundary of sampled multi-object transitions."""

from types import SimpleNamespace

import pytest

from agents.selectable_navigation_model import (
    MultiActorState,
    SelectableNavigationModel,
)
from agents.tile_scene_binding import TileActor, TileScene


def scene(positions, selected=0, warnings=()):
    actors = tuple(
        TileActor(f"actor-{i}", position, 0 if i == selected else 5, ())
        for i, position in enumerate(positions)
    )
    return TileScene(
        actors,
        None if selected is None else actors[selected].identity,
        (),
        frozenset(),
        frozenset(),
        warnings,
    )


def calibrated_model():
    initial = scene(((0, 0), (1, 0)))
    binding = SimpleNamespace(
        layout={"rows": 1, "cols": 9},
        floor_color=1,
        goals=((0, 0), (7, 0)),
        cell_pixel=lambda cell: cell,
        parse=lambda frame: initial,
    )
    model = SelectableNavigationModel(
        binding, SimpleNamespace(frame=[[[0, 5, 1, 15, 15, 1, 1, 1, 1]]])
    )
    # The input number is deliberately different from ka59's right action.
    model.learn(scene(((0, 0), (8, 0))), 2, scene(((1, 0), (8, 0))))
    return model


def test_observed_contact_moves_passive_object_and_preserves_selection():
    model = calibrated_model()
    model.learn(scene(((0, 0), (1, 0))), 2, scene(((0, 0), (6, 0))))
    state = MultiActorState(((0, 0), (1, 0)), 0)
    transitions = list(model.successors(state))
    assert ("move:2", MultiActorState(((0, 0), (6, 0)), 0), 1) in transitions
    assert ("select:1", MultiActorState(state.positions, 1), 1) in transitions
    assert not any(token == "move:4" for token, _, _ in transitions)


@pytest.mark.parametrize("receiver,landing", [((2, 0), (7, 0)), ((1, 0), (0, 0))])
def test_noncontact_or_opposite_displacement_does_not_become_rule(receiver, landing):
    model = calibrated_model()
    model.learn(scene(((0, 0), receiver)), 2, scene(((0, 0), landing)))
    assert not model.contact_rules


@pytest.mark.parametrize("violation", ["different_terrain", "occupied_landing"])
def test_observed_relocation_is_not_reused_without_its_preconditions(violation):
    model = calibrated_model()
    model.learn(scene(((0, 0), (1, 0))), 2, scene(((0, 0), (6, 0))))
    positions = ((0, 0), (1, 0))
    if violation == "different_terrain":
        model.terrain[(3, 0)] = 1
    else:
        positions += ((6, 0),)
    assert not any(
        token == "move:2"
        for token, _, _ in model.successors(MultiActorState(positions, 0))
    )


def test_uncertainty_and_contradictory_direction_stop_planning():
    model = calibrated_model()
    with pytest.raises(ValueError, match="uncertain identities or selection"):
        model.state(
            scene(((0, 0), (1, 0)), warnings=("ambiguous_identity_assignment",))
        )
    with pytest.raises(ValueError, match="uncertain identities or selection"):
        model.state(scene(((0, 0), (1, 0)), selected=None))
    with pytest.raises(ValueError, match="direction model contradicted"):
        model.learn(scene(((1, 0), (8, 0))), 2, scene(((0, 0), (8, 0))))

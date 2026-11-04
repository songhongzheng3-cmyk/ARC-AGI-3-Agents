from types import SimpleNamespace

from agents.mechanism_state_tracker import MechanismStateTracker, pattern_key


def _frame(grid, level=0):
    return SimpleNamespace(
        frame=[grid],
        levels_completed=level,
        state=SimpleNamespace(name="NOT_FINISHED"),
    )


def _synthetic_grid(resource_width=8, status_variant=0, status_color=9):
    grid = [[4 for _ in range(64)] for _ in range(64)]
    for y in range(5, 55):
        for x in range(9, 59):
            grid[y][x] = 3
    for y in range(53, 61):
        for x in range(1, 11):
            grid[y][x] = 5
    pattern = (
        ((1, 1, 1), (1, 0, 0), (1, 0, 1))
        if status_variant == 0
        else ((1, 1, 1), (0, 0, 1), (1, 0, 1))
    )
    for row, values in enumerate(pattern):
        for col, value in enumerate(values):
            if value:
                for yy in range(55 + row * 2, 57 + row * 2):
                    for xx in range(3 + col * 2, 5 + col * 2):
                        grid[yy][xx] = status_color
    for y in (61, 62):
        for x in range(55 - resource_width, 55):
            grid[y][x] = 11
    return grid


def test_extracts_layout_resource_and_status_pattern():
    tracker = MechanismStateTracker()
    state = tracker.observe(_frame(_synthetic_grid()))

    assert state.layout is not None
    assert state.layout.origin == (9, 5)
    assert state.layout.cell_size == (5, 5)
    assert (state.layout.cols, state.layout.rows) == (10, 10)
    assert state.resource_level == 16
    assert state.resource_capacity == 16
    assert state.status_signature == "111/100/101"


def test_emits_mechanism_and_level_events():
    tracker = MechanismStateTracker()
    tracker.observe(_frame(_synthetic_grid(resource_width=4)))
    refill = tracker.observe(
        _frame(_synthetic_grid(resource_width=8)), action="ACTION1"
    )
    changed = tracker.observe(
        _frame(_synthetic_grid(resource_width=7, status_variant=1)),
        action="ACTION2",
    )
    advanced = tracker.observe(
        _frame(_synthetic_grid(resource_width=6, status_variant=1), level=1),
        action="ACTION3",
    )

    assert "resource_increased" in {event.event_type for event in refill.events}
    assert "status_pattern_changed" in {event.event_type for event in changed.events}
    assert "level_advanced" in {event.event_type for event in advanced.events}
    assert refill.task_signal_observed
    assert pattern_key(changed.status_pattern) == "111/001/101"


def test_empty_resource_meter_is_not_a_full_meter_in_the_background_color():
    tracker = MechanismStateTracker()
    tracker.observe(_frame(_synthetic_grid(resource_width=8)))
    depleted = _synthetic_grid(resource_width=0)
    for y in (61, 62):
        for x in range(47, 55):
            depleted[y][x] = 3
    empty = tracker.observe(_frame(depleted), action="ACTION1")
    assert empty.resource_level == 0
    assert empty.resource_capacity == 16
    assert "resource_decreased" in {event.event_type for event in empty.events}
    assert "resource_increased" not in {event.event_type for event in empty.events}
    refill = tracker.observe(_frame(_synthetic_grid()), action="ACTION1")
    assert refill.resource_level == 16


def test_resource_palette_is_reinferred_when_level_changes():
    tracker = MechanismStateTracker()
    tracker.observe(_frame(_synthetic_grid()))
    recolored = _synthetic_grid()
    for y in (61, 62):
        for x in range(47, 55):
            recolored[y][x] = 12
    state = tracker.observe(_frame(recolored, level=1))
    assert state.resource_level == 16
    assert tracker.resource_color == 12


def test_emits_palette_event_when_pattern_is_unchanged():
    tracker = MechanismStateTracker()
    before = tracker.observe(_frame(_synthetic_grid(status_color=9)))
    after = tracker.observe(
        _frame(_synthetic_grid(status_color=12)),
        action="ACTION2",
    )

    assert before.status_pattern == after.status_pattern
    assert before.status_color == 9
    assert after.status_color == 12
    assert "status_palette_changed" in {event.event_type for event in after.events}


def test_locks_layout_until_level_changes():
    tracker = MechanismStateTracker()
    first = tracker.observe(_frame(_synthetic_grid()))
    changed_grid = _synthetic_grid()
    for y in range(5, 25):
        for x in range(9, 24):
            changed_grid[y][x] = 4

    same_level = tracker.observe(_frame(changed_grid), action="ACTION4")
    next_level = tracker.observe(_frame(changed_grid, level=1), action="ACTION4")

    assert same_level.layout == first.layout
    assert next_level.layout is not first.layout
    assert "level_advanced" in {event.event_type for event in next_level.events}


def test_resolved_player_rejects_unrelated_translated_patch():
    tracker = MechanismStateTracker()
    before = _synthetic_grid()
    after = _synthetic_grid()
    tracker.observe(_frame(before))
    tracker.player_position = (1, 1)
    tracker.player_confidence = 0.95
    # Move an unrelated two-colour patch far away from the expected ACTION4
    # destination; continuity must prevent a false player teleport.
    before[20][30] = 8
    before[20][31] = 9
    after[20][30] = 3
    after[20][31] = 3
    after[20][35] = 8
    after[20][36] = 9
    tracker.previous_grid = __import__("numpy").asarray(before)

    state = tracker.observe(_frame(after), action="ACTION4")

    assert state.player_position == (2, 1)


def test_detects_unique_nonlocal_player_relocation():
    tracker = MechanismStateTracker()
    before = _synthetic_grid()
    after = _synthetic_grid()
    sprite = [
        [12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
    ]
    for row, values in enumerate(sprite):
        before[10 + row][9:14] = values
        after[5 + row][34:39] = values
    tracker.observe(_frame(before))
    tracker.player_position = (0, 1)
    tracker.player_confidence = 0.95
    tracker.previous_state = tracker.previous_state.__class__(
        **{
            **tracker.previous_state.__dict__,
            "player_position": (0, 1),
            "player_confidence": 0.95,
        }
    )

    state = tracker.observe(_frame(after), action="ACTION1")

    assert state.player_position == (5, 0)
    relocation = [
        event for event in state.events if event.event_type == "player_relocated"
    ]
    assert len(relocation) == 1
    assert relocation[0].before == (0, 1)
    assert relocation[0].after == (5, 0)


def test_emits_object_lifecycle_events_for_changed_cell_group():
    tracker = MechanismStateTracker()
    before = _synthetic_grid()
    after = _synthetic_grid()
    for y in range(20, 23):
        for x in range(20, 23):
            before[y][x] = 8
            after[y][x] = 3

    tracker.observe(_frame(before))
    state = tracker.observe(_frame(after), action="ACTION4")

    disappeared = [
        event for event in state.events if event.event_type == "object_disappeared"
    ]
    assert disappeared
    assert any(event.cells == ((2, 3),) for event in disappeared)


def test_marks_stacked_full_width_bands_as_player_components():
    tracker = MechanismStateTracker()
    grid = _synthetic_grid()
    # A two-colour 5x5 sprite whose connected colour bands are parsed as two
    # objects in adjacent logical rows.
    for y in range(30, 32):
        for x in range(34, 39):
            grid[y][x] = 12
    for y in range(32, 35):
        for x in range(34, 39):
            grid[y][x] = 9
    tracker.observe(_frame(grid))
    tracker.player_position = (5, 5)
    tracker.player_confidence = 0.95

    state = tracker.observe(_frame(grid), action="ACTION2")

    components = [
        item
        for item in state.objects
        if item.object_id in state.player_component_ids
    ]
    assert {(item.color, item.area) for item in components} == {(12, 10), (9, 15)}


def test_resolves_full_cell_composite_player_from_commanded_translation():
    tracker = MechanismStateTracker()
    before = _synthetic_grid()
    after = _synthetic_grid()
    for y in range(30, 35):
        for x in range(34, 39):
            before[y][x] = 12 if y < 32 else 9
            after[y][x] = 3
            after[y][x + 5] = 12 if y < 32 else 9

    tracker.observe(_frame(before))
    state = tracker.observe(_frame(after), action="ACTION4")

    assert state.player_position == (6, 5)
    assert state.player_confidence == 0.99
    assert len(state.player_component_ids) == 2


def test_compact_objects_share_composite_player_lattice_coordinates():
    tracker = MechanismStateTracker()
    grid = _synthetic_grid()
    for y in range(16, 19):
        for x in range(20, 23):
            grid[y][x] = 11
    grid[17][21] = 3

    state = tracker.observe(_frame(grid))
    ring = next(item for item in state.objects if item.color == 11 and item.area == 8)

    assert ring.cell == (2, 2)

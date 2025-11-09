"""Tests for expert-goal linking functionality.

This tests the integration between the Goals/Directive system and the Dynamic Subbrains
Expert system, ensuring goals can be properly linked to and managed with experts.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aios.core.directives import (
    add_directive,
    list_directives,
    link_directive_to_expert,
    unlink_directive_from_expert,
    get_directives_for_expert,
    get_experts_for_directive,
    _ensure_expert_id_column,
)
from aios.memory.store import init_db


@pytest.fixture
def db_conn():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    init_db(conn)
    
    yield conn
    
    conn.close()
    Path(db_path).unlink(missing_ok=True)


def test_ensure_expert_id_column_creates_column(db_conn):
    """Test that migration creates expert_id column."""
    # First, verify the column gets created
    _ensure_expert_id_column(db_conn)
    
    cursor = db_conn.execute("PRAGMA table_info(directives)")
    columns = [row[1] for row in cursor.fetchall()]
    
    assert "expert_id" in columns


def test_ensure_expert_id_column_idempotent(db_conn):
    """Test that migration can be called multiple times safely."""
    _ensure_expert_id_column(db_conn)
    _ensure_expert_id_column(db_conn)  # Should not raise
    
    cursor = db_conn.execute("PRAGMA table_info(directives)")
    columns = [row[1] for row in cursor.fetchall()]
    
    assert "expert_id" in columns


def test_add_directive_without_expert(db_conn):
    """Test adding a directive without expert linkage."""
    directive_id = add_directive(db_conn, "Learn Python programming")
    
    assert directive_id > 0
    
    directives = list_directives(db_conn, active_only=True)
    assert len(directives) == 1
    assert directives[0].text == "Learn Python programming"
    assert directives[0].expert_id is None


def test_add_directive_with_expert(db_conn):
    """Test adding a directive with expert linkage."""
    expert_id = "python_expert"
    directive_id = add_directive(db_conn, "Learn Python programming", expert_id=expert_id)
    
    assert directive_id > 0
    
    directives = list_directives(db_conn, active_only=True)
    assert len(directives) == 1
    assert directives[0].text == "Learn Python programming"
    assert directives[0].expert_id == expert_id


def test_list_directives_filter_by_expert(db_conn):
    """Test filtering directives by expert_id."""
    # Add directives for different experts
    add_directive(db_conn, "Learn Python", expert_id="python_expert")
    add_directive(db_conn, "Learn Calculus", expert_id="math_expert")
    add_directive(db_conn, "General goal")  # No expert
    
    # Filter by python expert
    python_goals = list_directives(db_conn, active_only=True, expert_id="python_expert")
    assert len(python_goals) == 1
    assert python_goals[0].text == "Learn Python"
    assert python_goals[0].expert_id == "python_expert"
    
    # Filter by math expert
    math_goals = list_directives(db_conn, active_only=True, expert_id="math_expert")
    assert len(math_goals) == 1
    assert math_goals[0].text == "Learn Calculus"
    assert math_goals[0].expert_id == "math_expert"
    
    # Get all
    all_goals = list_directives(db_conn, active_only=True)
    assert len(all_goals) == 3


def test_link_directive_to_expert(db_conn):
    """Test linking an existing directive to an expert."""
    # Add directive without expert
    directive_id = add_directive(db_conn, "Learn Python")
    
    # Verify no expert initially
    directives = list_directives(db_conn, active_only=True)
    assert directives[0].expert_id is None
    
    # Link to expert
    success = link_directive_to_expert(db_conn, directive_id, "python_expert")
    assert success is True
    
    # Verify linkage
    directives = list_directives(db_conn, active_only=True)
    assert directives[0].expert_id == "python_expert"


def test_link_directive_to_expert_invalid_id(db_conn):
    """Test linking with invalid directive ID."""
    success = link_directive_to_expert(db_conn, 99999, "python_expert")
    assert success is False


def test_unlink_directive_from_expert(db_conn):
    """Test removing expert association from directive."""
    # Add directive with expert
    directive_id = add_directive(db_conn, "Learn Python", expert_id="python_expert")
    
    # Verify expert linkage
    directives = list_directives(db_conn, active_only=True)
    assert directives[0].expert_id == "python_expert"
    
    # Unlink
    success = unlink_directive_from_expert(db_conn, directive_id)
    assert success is True
    
    # Verify no linkage
    directives = list_directives(db_conn, active_only=True)
    assert directives[0].expert_id is None


def test_unlink_directive_invalid_id(db_conn):
    """Test unlinking with invalid directive ID."""
    success = unlink_directive_from_expert(db_conn, 99999)
    assert success is False


def test_get_directives_for_expert(db_conn):
    """Test getting all directives for a specific expert."""
    # Add multiple directives for same expert
    add_directive(db_conn, "Learn Python basics", expert_id="python_expert")
    add_directive(db_conn, "Master Python advanced", expert_id="python_expert")
    add_directive(db_conn, "Learn Calculus", expert_id="math_expert")
    
    # Get Python expert goals
    python_goals = get_directives_for_expert(db_conn, "python_expert")
    assert len(python_goals) == 2
    assert all(d.expert_id == "python_expert" for d in python_goals)
    
    # Get Math expert goals
    math_goals = get_directives_for_expert(db_conn, "math_expert")
    assert len(math_goals) == 1
    assert math_goals[0].expert_id == "math_expert"


def test_get_directives_for_expert_no_match(db_conn):
    """Test getting directives for expert with no linked goals."""
    add_directive(db_conn, "Learn Python", expert_id="python_expert")
    
    # Query for different expert
    goals = get_directives_for_expert(db_conn, "nonexistent_expert")
    assert len(goals) == 0


def test_get_experts_for_directive(db_conn):
    """Test getting expert ID for a directive."""
    # Add directive with expert
    directive_id = add_directive(db_conn, "Learn Python", expert_id="python_expert")
    
    # Get expert
    expert_id = get_experts_for_directive(db_conn, directive_id)
    assert expert_id == "python_expert"


def test_get_experts_for_directive_no_link(db_conn):
    """Test getting expert for directive without linkage."""
    # Add directive without expert
    directive_id = add_directive(db_conn, "General goal")
    
    # Get expert (should be None)
    expert_id = get_experts_for_directive(db_conn, directive_id)
    assert expert_id is None


def test_get_experts_for_directive_invalid_id(db_conn):
    """Test getting expert for invalid directive ID."""
    expert_id = get_experts_for_directive(db_conn, 99999)
    assert expert_id is None


def test_backward_compatibility_old_directives(db_conn):
    """Test that old directives (created before expert_id) still work."""
    # Manually insert directive without expert_id column awareness
    cursor = db_conn.execute(
        "INSERT INTO directives(text, active) VALUES (?, 1)",
        ("Old directive",),
    )
    db_conn.commit()
    directive_id = cursor.lastrowid
    
    # Ensure migration runs
    _ensure_expert_id_column(db_conn)
    
    # Should be able to list it
    directives = list_directives(db_conn, active_only=True)
    assert len(directives) == 1
    assert directives[0].text == "Old directive"
    assert directives[0].expert_id is None
    
    # Should be able to link it
    success = link_directive_to_expert(db_conn, directive_id, "new_expert")
    assert success is True
    
    # Verify linkage
    expert_id = get_experts_for_directive(db_conn, directive_id)
    assert expert_id == "new_expert"


def test_workflow_auto_training_integration(db_conn):
    """Test workflow: Auto-training creates expert and links goal."""
    # Simulate auto-training workflow
    user_message = "Learn Python programming"
    expert_id = "python_expert_001"
    
    # 1. User provides learning intent
    # 2. System creates directive with expert linkage
    directive_id = add_directive(db_conn, user_message, expert_id=expert_id)
    
    # 3. Verify linkage exists
    linked_expert = get_experts_for_directive(db_conn, directive_id)
    assert linked_expert == expert_id
    
    # 4. Query all goals for this expert
    expert_goals = get_directives_for_expert(db_conn, expert_id)
    assert len(expert_goals) == 1
    assert expert_goals[0].text == user_message
    
    # 5. Later: User removes goal
    unlink_directive_from_expert(db_conn, directive_id)
    
    # 6. Verify unlinked
    linked_expert = get_experts_for_directive(db_conn, directive_id)
    assert linked_expert is None


def test_workflow_multiple_goals_per_expert(db_conn):
    """Test workflow: Multiple goals linked to same expert."""
    expert_id = "python_expert"
    
    # Add multiple related goals
    goal1_id = add_directive(db_conn, "Learn Python basics", expert_id=expert_id)
    goal2_id = add_directive(db_conn, "Master Python OOP", expert_id=expert_id)
    goal3_id = add_directive(db_conn, "Understand Python async", expert_id=expert_id)
    
    # Get all goals for expert
    goals = get_directives_for_expert(db_conn, expert_id)
    assert len(goals) == 3
    
    # Verify all have same expert
    for goal in goals:
        assert goal.expert_id == expert_id
    
    # Remove one goal's linkage
    unlink_directive_from_expert(db_conn, goal2_id)
    
    # Should have 2 linked goals now
    goals = get_directives_for_expert(db_conn, expert_id)
    assert len(goals) == 2
    assert goal2_id not in [g.directive_id for g in goals]


def test_workflow_expert_reassignment(db_conn):
    """Test workflow: Reassigning goal from one expert to another."""
    # Create goal linked to first expert
    directive_id = add_directive(db_conn, "Learn programming", expert_id="general_expert")
    
    # Verify initial linkage
    expert_id = get_experts_for_directive(db_conn, directive_id)
    assert expert_id == "general_expert"
    
    # Reassign to specialized expert
    success = link_directive_to_expert(db_conn, directive_id, "python_expert")
    assert success is True
    
    # Verify new linkage
    expert_id = get_experts_for_directive(db_conn, directive_id)
    assert expert_id == "python_expert"
    
    # Verify old expert has no goals
    old_goals = get_directives_for_expert(db_conn, "general_expert")
    assert len(old_goals) == 0
    
    # Verify new expert has the goal
    new_goals = get_directives_for_expert(db_conn, "python_expert")
    assert len(new_goals) == 1
    assert new_goals[0].directive_id == directive_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Validation tests for the 2PC implementation."""
from two_phase_commit import TwoPhaseCommitSystem


def test_successful_commit():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2", "db3"])
    result = system.execute({
        "db1": [{"op": "set", "key": "account_A", "value": 900}],
        "db2": [{"op": "set", "key": "account_B", "value": 1100}],
    })
    assert result["outcome"] == "committed"
    assert system.participants["db1"].get("account_A") == 900
    assert system.participants["db2"].get("account_B") == 1100
    print("PASS: successful commit")


def test_abort_on_unavailable():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    system.participants["db2"].set_available(False)
    result = system.execute({
        "db1": [{"op": "set", "key": "x", "value": 1}],
        "db2": [{"op": "set", "key": "y", "value": 2}],
    })
    assert result["outcome"] == "aborted"
    assert result["reason"] is not None
    assert system.participants["db1"].get("x") is None
    print("PASS: abort on unavailable participant")


def test_abort_no_state_change():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    system.execute({"db1": [{"op": "set", "key": "pre", "value": "exists"}]})
    system.participants["db2"].set_available(False)
    result = system.execute({
        "db1": [{"op": "set", "key": "pre", "value": "changed"}],
        "db2": [{"op": "set", "key": "y", "value": 2}],
    })
    assert result["outcome"] == "aborted"
    assert system.participants["db1"].get("pre") == "exists"
    print("PASS: aborted transactions don't modify state")


def test_committed_durable():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    system.execute({
        "db1": [{"op": "set", "key": "a", "value": 1}],
        "db2": [{"op": "set", "key": "b", "value": 2}],
    })
    assert system.participants["db1"].get("a") == 1
    assert system.participants["db2"].get("b") == 2
    print("PASS: committed transactions durable")


def test_lock_conflict():
    system = TwoPhaseCommitSystem(participant_ids=["db1"])
    p = system.participants["db1"]
    p.prepare("manual-tx", [{"op": "set", "key": "locked_key", "value": 42}])
    result = system.execute({"db1": [{"op": "set", "key": "locked_key", "value": 99}]})
    assert result["outcome"] == "aborted"
    p.abort("manual-tx")
    print("PASS: lock conflict aborts")


def test_locks_released_after_commit():
    system = TwoPhaseCommitSystem(participant_ids=["db1"])
    system.execute({"db1": [{"op": "set", "key": "k", "value": 1}]})
    result = system.execute({"db1": [{"op": "set", "key": "k", "value": 2}]})
    assert result["outcome"] == "committed"
    assert system.participants["db1"].get("k") == 2
    print("PASS: locks released after commit")


def test_locks_released_after_abort():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    system.participants["db2"].set_available(False)
    system.execute({
        "db1": [{"op": "set", "key": "k", "value": 1}],
        "db2": [{"op": "set", "key": "j", "value": 1}],
    })
    system.participants["db2"].set_available(True)
    result = system.execute({"db1": [{"op": "set", "key": "k", "value": 2}]})
    assert result["outcome"] == "committed"
    print("PASS: locks released after abort")


def test_delete_operation():
    system = TwoPhaseCommitSystem(participant_ids=["db1"])
    system.execute({"db1": [{"op": "set", "key": "del_me", "value": "temp"}]})
    assert system.participants["db1"].get("del_me") == "temp"
    result = system.execute({"db1": [{"op": "delete", "key": "del_me"}]})
    assert result["outcome"] == "committed"
    assert system.participants["db1"].get("del_me") is None
    print("PASS: delete operation")


def test_coordinator_recovery():
    system = TwoPhaseCommitSystem(participant_ids=["n1", "n2"])
    tx = system.coordinator.begin_transaction()
    system.participants["n1"].prepare(tx, [{"op": "set", "key": "a", "value": 1}])
    system.participants["n2"].prepare(tx, [{"op": "set", "key": "b", "value": 2}])
    system.coordinator.log.append({"tx_id": tx, "state": "preparing", "participants": ["n1", "n2"]})
    system.coordinator.log.append({"tx_id": tx, "state": "committing"})
    result = system.coordinator.recover()
    assert tx in result["recovered_transactions"]
    assert result["decisions_resent"] == 2
    assert system.participants["n1"].get("a") == 1
    assert system.participants["n2"].get("b") == 2
    print("PASS: coordinator recovery")


def test_participant_recovery():
    system = TwoPhaseCommitSystem(participant_ids=["p1"])
    p1 = system.participants["p1"]
    p1.prepare("tx-orphan", [{"op": "set", "key": "z", "value": 99}])
    in_doubt = p1.recover()
    assert "tx-orphan" in in_doubt
    print("PASS: participant recovery identifies in-doubt txns")


def test_transaction_log():
    system = TwoPhaseCommitSystem(participant_ids=["a", "b"])
    system.execute({"a": [{"op": "set", "key": "x", "value": 1}]})
    states = [e["state"] for e in system.coordinator.log]
    assert "initiated" in states
    assert "preparing" in states
    assert "committed" in states
    print("PASS: transaction log records transitions")


def test_concurrent_different_keys():
    system = TwoPhaseCommitSystem(participant_ids=["db1"])
    r1 = system.execute({"db1": [{"op": "set", "key": "k1", "value": "v1"}]})
    r2 = system.execute({"db1": [{"op": "set", "key": "k2", "value": "v2"}]})
    assert r1["outcome"] == "committed"
    assert r2["outcome"] == "committed"
    print("PASS: concurrent transactions on different keys")


def test_timeout_unavailable():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    system.participants["db2"].set_available(False)
    result = system.execute({
        "db1": [{"op": "set", "key": "a", "value": 1}],
        "db2": [{"op": "set", "key": "b", "value": 2}],
    })
    assert result["outcome"] == "aborted"
    assert "timeout" in result["reason"] or "unavailable" in result["reason"]
    print("PASS: timeout on unavailable participant")


def test_full_lifecycle():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    tx_id = system.coordinator.begin_transaction()
    result = system.coordinator.execute_transaction(tx_id, {
        "db1": [{"op": "set", "key": "life", "value": "cycle"}],
        "db2": [{"op": "set", "key": "full", "value": "test"}],
    })
    assert result["outcome"] == "committed"
    assert system.participants["db1"].get("life") == "cycle"
    assert system.coordinator.get_transaction_state(tx_id) == "committed"
    print("PASS: full lifecycle (begin, prepare, commit, verify)")


def test_get_all_states():
    system = TwoPhaseCommitSystem(participant_ids=["db1", "db2"])
    system.execute({"db1": [{"op": "set", "key": "x", "value": 1}]})
    s = system.get_all_states()
    assert "db1" in s and "db2" in s
    assert s["db1"]["x"] == 1
    print("PASS: get_all_states")


if __name__ == "__main__":
    test_successful_commit()
    test_abort_on_unavailable()
    test_abort_no_state_change()
    test_committed_durable()
    test_lock_conflict()
    test_locks_released_after_commit()
    test_locks_released_after_abort()
    test_delete_operation()
    test_coordinator_recovery()
    test_participant_recovery()
    test_transaction_log()
    test_concurrent_different_keys()
    test_timeout_unavailable()
    test_full_lifecycle()
    test_get_all_states()
    print("\nAll 15 tests passed!")

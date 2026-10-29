from ticket_to_code.verification.oscillation_detector import (
    OscillationDetector,
    NormalizedDiagnostic,
    VerificationTransition,
)

def test_compound_state_key_oscillation_detection():
    detector = OscillationDetector(max_recovery_turns=3)

    d1 = NormalizedDiagnostic.create("app.ts", "TS2304", "Cannot find name 'User'")
    d2 = NormalizedDiagnostic.create("app.ts", "TS2307", "Cannot find module 'lodash'")

    # Turn 1: baseline with diff A
    eval1 = detector.evaluate([d1], diff_hash="DIFF_A")
    assert eval1.transition == VerificationTransition.PROGRESS
    assert eval1.should_continue is True

    # Turn 2: progress with diff B
    eval2 = detector.evaluate([d2], diff_hash="DIFF_B")
    assert eval2.transition == VerificationTransition.REGRESSION or eval2.transition == VerificationTransition.PROGRESS

    # Turn 3: patch reverts to exact DIFF_A and introduces exact d1 again -> state cycle!
    eval3 = detector.evaluate([d1], diff_hash="DIFF_A")
    assert eval3.transition == VerificationTransition.OSCILLATION
    assert eval3.should_continue is False
    assert eval3.requires_rollback is True
    assert "Oscillation cycle detected" in eval3.diagnostic_summary or "Oscillation" in eval3.diagnostic_summary

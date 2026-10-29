package com.cloudforge.commerce.settlement.model;

public enum SettlementStatus {
    PENDING,
    RECONCILED,
    DISCREPANCY_FLAGGED,
    SETTLED,
    FAILED;

    public boolean isSettled() {
        return this == SETTLED;
    }
}
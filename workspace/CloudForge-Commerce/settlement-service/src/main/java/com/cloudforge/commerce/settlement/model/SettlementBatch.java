package com.cloudforge.commerce.settlement.model;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.Instant;

@Entity
@Table(name = "settlement_batches")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SettlementBatch {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "batch_number", nullable = false, unique = true)
    private String batchNumber;

    @Column(name = "merchant_id", nullable = false)
    private String merchantId;

    @Column(name = "gross_settlement_amount", nullable = false)
    private BigDecimal grossSettlementAmount;

    @Column(name = "net_payout_amount", nullable = false)
    private BigDecimal netPayoutAmount;

    @Column(name = "variance_amount", nullable = false)
    private BigDecimal varianceAmount;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private SettlementStatus status;

    @Column(name = "reconciliation_note")
    private String reconciliationNote;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "settled_at")
    private Instant settledAt;

    @PrePersist
    protected void onCreate() {
        this.createdAt = Instant.now();
        if (this.varianceAmount == null) {
            this.varianceAmount = BigDecimal.ZERO;
        }
    }

    public void ensureNotSettled() {
        if (status == SettlementStatus.SETTLED) {
            throw new IllegalStateException("Cannot reconcile a finalized settlement batch");
        }
    }
}

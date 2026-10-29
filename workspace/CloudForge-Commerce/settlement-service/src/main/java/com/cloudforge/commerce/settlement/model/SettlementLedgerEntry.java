package com.cloudforge.commerce.settlement.model;

import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.Instant;

@Entity
@Table(name = "settlement_ledger_entries")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SettlementLedgerEntry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "batch_id", nullable = false)
    private Long batchId;

    @Column(name = "order_number", nullable = false)
    private String orderNumber;

    @Column(name = "reservation_id", nullable = false)
    private Long reservationId;

    @Column(name = "authorized_amount", nullable = false)
    private BigDecimal authorizedAmount;

    @Column(name = "settled_amount", nullable = false)
    private BigDecimal settledAmount;

    @Column(name = "variance_delta", nullable = false)
    private BigDecimal varianceDelta;

    @Column(name = "is_orphaned_hold", nullable = false)
    private Boolean isOrphanedHold;

    @Column(name = "recorded_at", nullable = false, updatable = false)
    private Instant recordedAt;

    @PrePersist
    protected void onCreate() {
        this.recordedAt = Instant.now();
        if (this.isOrphanedHold == null) {
            this.isOrphanedHold = false;
        }
        if (this.varianceDelta == null) {
            this.varianceDelta = BigDecimal.ZERO;
        }
    }
}

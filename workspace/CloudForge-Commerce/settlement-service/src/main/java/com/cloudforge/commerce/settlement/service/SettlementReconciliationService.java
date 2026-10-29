package com.cloudforge.commerce.settlement.service;

import com.cloudforge.commerce.settlement.model.SettlementBatch;
import com.cloudforge.commerce.settlement.model.SettlementLedgerEntry;
import com.cloudforge.commerce.settlement.model.SettlementStatus;
import com.cloudforge.commerce.settlement.repository.SettlementBatchRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class SettlementReconciliationService {

    private final SettlementBatchRepository settlementBatchRepository;

    @Transactional
    public SettlementBatch createBatch(String batchNumber, String merchantId, BigDecimal grossAmount) {
        log.info("Creating settlement batch {} for merchant {}", batchNumber, merchantId);
        SettlementBatch batch = SettlementBatch.builder()
                .batchNumber(batchNumber)
                .merchantId(merchantId)
                .grossSettlementAmount(grossAmount)
                .netPayoutAmount(grossAmount)
                .varianceAmount(BigDecimal.ZERO)
                .status(SettlementStatus.PENDING)
                .build();
        return settlementBatchRepository.save(batch);
    }

    @Transactional
    public SettlementBatch reconcileBatch(Long batchId, List<SettlementLedgerEntry> entries) {
        log.info("Reconciling settlement batch {}", batchId);
        SettlementBatch batch = settlementBatchRepository.findById(batchId)
                .orElseThrow(() -> new IllegalArgumentException("Settlement batch not found: " + batchId));

        batch.ensureNotSettled();

        BigDecimal totalSettled = BigDecimal.ZERO;
        boolean hasOrphanedHold = false;

        for (SettlementLedgerEntry entry : entries) {
            totalSettled = totalSettled.add(entry.getSettledAmount());
            if (Boolean.TRUE.equals(entry.getIsOrphanedHold())) {
                hasOrphanedHold = true;
                log.warn("Detected orphaned active hold in settlement batch {} for reservation {}", 
                        batchId, entry.getReservationId());
            }
        }

        BigDecimal variance = batch.getGrossSettlementAmount().subtract(totalSettled);
        batch.setVarianceAmount(variance);
        batch.setNetPayoutAmount(totalSettled);

        if (hasOrphanedHold || variance.compareTo(BigDecimal.ZERO) != 0) {
            batch.setStatus(SettlementStatus.DISCREPANCY_FLAGGED);
            batch.setReconciliationNote("Variance detected or orphaned hold detected: delta=" + variance);
        } else {
            batch.setStatus(SettlementStatus.RECONCILED);
            batch.setSettledAt(Instant.now());
            batch.setReconciliationNote("Settlement 100% Consistent - Zero Discrepancies");
        }

        return settlementBatchRepository.save(batch);
    }
}

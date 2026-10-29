package com.cloudforge.commerce.settlement.repository;

import com.cloudforge.commerce.settlement.model.SettlementBatch;
import com.cloudforge.commerce.settlement.model.SettlementStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SettlementBatchRepository extends JpaRepository<SettlementBatch, Long> {
    Optional<SettlementBatch> findByBatchNumber(String batchNumber);
    List<SettlementBatch> findByMerchantId(String merchantId);
    List<SettlementBatch> findByStatus(SettlementStatus status);
}

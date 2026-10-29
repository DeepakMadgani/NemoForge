package com.cloudforge.commerce.order.repository;

import com.cloudforge.commerce.order.model.PaymentReservation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PaymentReservationRepository extends JpaRepository<PaymentReservation, Long> {
    List<PaymentReservation> findByCustomerId(String customerId);
}

package com.cloudforge.commerce.order.service;

import com.cloudforge.commerce.order.model.PaymentReservation;
import com.cloudforge.commerce.order.model.ReservationStatus;
import com.cloudforge.commerce.order.repository.PaymentReservationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;

@Service
@RequiredArgsConstructor
@Slf4j
public class PaymentLedgerService {

    private final PaymentReservationRepository reservationRepository;

    @Transactional
    public PaymentReservation createReservation(String customerId, BigDecimal amount) {
        log.info("Creating payment reservation for customer {} amount {}", customerId, amount);
        PaymentReservation reservation = PaymentReservation.builder()
                .customerId(customerId)
                .reservedAmount(amount)
                .status(ReservationStatus.ACTIVE)
                .build();
        return reservationRepository.save(reservation);
    }

    @Transactional
    public void releaseReservation(Long reservationId, String reason) {
        log.info("Releasing payment reservation {} reason: {}", reservationId, reason);
        if (reservationId == null) {
            log.warn("Cannot release null reservation id");
            return;
        }
        PaymentReservation reservation = reservationRepository.findById(reservationId)
                .orElseThrow(() -> new IllegalArgumentException("Reservation not found: " + reservationId));
        
        reservation.setStatus(ReservationStatus.RELEASED);
        reservation.setReleaseReason(reason);
        reservationRepository.save(reservation);
        log.info("Payment reservation {} successfully released", reservationId);
    }

    @Transactional(readOnly = true)
    public ReservationStatus getReservationStatus(Long reservationId) {
        return reservationRepository.findById(reservationId)
                .map(PaymentReservation::getStatus)
                .orElse(null);
    }
}

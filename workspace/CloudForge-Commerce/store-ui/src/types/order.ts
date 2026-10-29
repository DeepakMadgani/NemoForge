export type OrderStatus = 'PENDING' | 'CONFIRMED' | 'FULFILLED' | 'CANCELLED' | 'REFUNDED';

export type ReservationStatus = 'HOLD_PENDING' | 'ACTIVE' | 'RELEASED' | 'SETTLED';

export interface Order {
  id: number;
  orderNumber: string;
  customerId: string;
  totalAmount: number;
  status: OrderStatus;
  reservationId?: number;
  createdAt: string;
  updatedAt: string;
}

export interface PaymentReservation {
  id: number;
  customerId: string;
  reservedAmount: number;
  status: ReservationStatus;
  releaseReason?: string;
  createdAt: string;
  updatedAt: string;
}

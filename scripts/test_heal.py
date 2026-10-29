import sys
from pathlib import Path
sys.path.insert(0, str(Path("nemoforge/src").resolve()))

from ticket_to_code.agents.code_generator import _heal_react_destructuring

sample = """
import React, { useState } from 'react';
import { Order, OrderStatus } from '../types/order';

interface OrderManagementProps {
  orders: Order[];
  onCancelOrder: (orderId: number, reason: string) => Promise<void>;
  onRefundOrder: (orderId: number) => Promise<void>;
}

export const OrderManagement: React.FC<OrderManagementProps> = ({ orders, onCancelOrder }) => {
  const [cancellingId, setCancellingId] = useState<number | null>(null);

  const handleRefund = async (orderId: number) => {
    await onRefundOrder(orderId);
  };

  return (
    <div>Test</div>
  );
};
"""

healed = _heal_react_destructuring(sample)
print("HEALED OUTPUT:")
print(healed)
assert "({ orders, onCancelOrder, onRefundOrder })" in healed
print("ASSERTION PASSED: onRefundOrder was automatically destructured!")


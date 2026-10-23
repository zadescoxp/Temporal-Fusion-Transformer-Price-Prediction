"""Transaction-cost assumptions applied to every back-test.

Commission is 10 bps per side and slippage is 5 bps per side.
A position that opens and later closes therefore pays 30 bps round-trip.
Costs are a fraction of notional, deducted from simple returns.
"""

COMMISSION_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
COST_PER_SIDE = COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE
ROUND_TRIP_COST = COST_PER_SIDE * 2

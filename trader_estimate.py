# Data from the GOLD report
# Non-Commercial positions
nc_long = 256077
nc_short = 61073
nc_spread = 50549
nc_long_count = 185
nc_short_count = 54
nc_spread_count = 77

# Commercial positions
c_long = 73323
c_short = 303883
c_long_count = 51
c_short_count = 51

# Total unique traders
total_unique = 261

# Calculate total positions for each category
nc_total_pos = nc_long + nc_short + nc_spread
c_total_pos = c_long + c_short

print('=== Position Data ===')
print(f'Non-Commercial total positions: {nc_total_pos:,} ({nc_long:,} long + {nc_short:,} short + {nc_spread:,} spread)')
print(f'Commercial total positions: {c_total_pos:,} ({c_long:,} long + {c_short:,} short)')
print()

# Position-based estimation
total_reportable_pos = nc_total_pos + c_total_pos
nc_share = nc_total_pos / total_reportable_pos

print('=== Position Shares ===')
print(f'Total reportable positions: {total_reportable_pos:,}')
print(f'Non-Commercial share: {nc_share:.1%}')
print(f'Commercial share: {1-nc_share:.1%}')
print()

# Initial estimates
estimated_nc_unique = total_unique * nc_share
estimated_c_unique = total_unique * (1 - nc_share)

print('=== Initial Position-Weighted Estimates ===')
print(f'Estimated Non-Commercial unique traders: {estimated_nc_unique:.1f}')
print(f'Estimated Commercial unique traders: {estimated_c_unique:.1f}')
print()

# Bounds checking for Non-Commercial
nc_min = max(nc_long_count, nc_short_count, nc_spread_count)
nc_max = min(nc_long_count + nc_short_count + nc_spread_count, total_unique)

print('=== Non-Commercial Bounds ===')
print(f'Minimum (max of individual counts): {nc_min}')
print(f'Maximum (sum of counts, capped by total): {nc_max} (sum would be {nc_long_count + nc_short_count + nc_spread_count})')
print()

# Constrain estimate within bounds
final_nc_estimate = max(min(estimated_nc_unique, nc_max), nc_min)
final_c_estimate = total_unique - final_nc_estimate

print('=== Final Constrained Estimates ===')
print(f'Non-Commercial unique traders: {final_nc_estimate:.0f}')
print(f'Commercial unique traders: {final_c_estimate:.0f}')
print(f'Total: {final_nc_estimate + final_c_estimate:.0f}')
print()

# Compare with simple count
print('=== Comparison with Simple Addition ===')
print(f'Non-Commercial (long + short + spread): {nc_long_count + nc_short_count + nc_spread_count}')
print(f'Commercial (long + short): {c_long_count + c_short_count}')
print(f'Simple total: {nc_long_count + nc_short_count + nc_spread_count + c_long_count + c_short_count}')
print(f'Actual unique total: {total_unique}')
print(f'Double counting: {nc_long_count + nc_short_count + nc_spread_count + c_long_count + c_short_count - total_unique} traders')
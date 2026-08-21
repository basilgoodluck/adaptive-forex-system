def scan_dc_events(prices: list[float], times: list, threshold: float):
    events = []
    extreme = prices[0]
    extreme_time = times[0]
    mode = "UPWARD"

    for i in range(1, len(prices)):
        p = prices[i]
        t = times[i]

        if mode == "UPWARD":
            if (p - extreme) / extreme >= threshold:
                events.append({
                    "type": "UP",
                    "time": t,
                    "price": p,
                    "extreme_price": extreme,
                    "extreme_time": extreme_time,
                })
                extreme = p
                extreme_time = t
                mode = "DOWNWARD"
            elif p < extreme:
                extreme = p
                extreme_time = t
        else:
            if (extreme - p) / extreme >= threshold:
                events.append({
                    "type": "DOWN",
                    "time": t,
                    "price": p,
                    "extreme_price": extreme,
                    "extreme_time": extreme_time,
                })
                extreme = p
                extreme_time = t
                mode = "UPWARD"
            elif p > extreme:
                extreme = p
                extreme_time = t

    return events

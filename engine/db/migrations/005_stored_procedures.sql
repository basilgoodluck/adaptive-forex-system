CREATE OR REPLACE FUNCTION compute_position_size(
    p_regime_label TEXT,
    p_atr DOUBLE PRECISION,
    p_atr_percentile DOUBLE PRECISION,
    p_confidence DOUBLE PRECISION,
    p_account_equity DOUBLE PRECISION,
    p_risk_fraction DOUBLE PRECISION,
    p_pip_value DOUBLE PRECISION,
    p_fixed_pip_fallback DOUBLE PRECISION,
    p_current_drawdown DOUBLE PRECISION,
    p_max_drawdown DOUBLE PRECISION,
    p_tp_ratio DOUBLE PRECISION
)
RETURNS TABLE (
    position_size DOUBLE PRECISION,
    sl_distance DOUBLE PRECISION,
    tp_distance DOUBLE PRECISION,
    used_atr BOOLEAN
) AS $$
DECLARE
    v_multiplier DOUBLE PRECISION;
    v_sl_distance DOUBLE PRECISION;
    v_position_size DOUBLE PRECISION;
    v_used_atr BOOLEAN;
BEGIN
    v_multiplier := CASE p_regime_label
        WHEN 'TREND_CALM' THEN 1.3
        WHEN 'TREND_VOLATILE' THEN 1.8
        WHEN 'RANGE_CALM' THEN 0.8
        WHEN 'RANGE_VOLATILE' THEN 1.2
        WHEN 'BREAKOUT' THEN 2.2
        ELSE 1.5
    END;

    IF p_atr_percentile IS NOT NULL AND (p_atr_percentile < 5 OR p_atr_percentile > 95) THEN
        v_sl_distance := p_fixed_pip_fallback;
        v_used_atr := FALSE;
    ELSE
        v_sl_distance := v_multiplier * p_atr;
        v_used_atr := TRUE;
    END IF;

    v_position_size := (p_account_equity * p_risk_fraction) / (v_sl_distance * p_pip_value);

    v_position_size := v_position_size * p_confidence;

    IF p_current_drawdown > (p_max_drawdown * 0.80) THEN
        v_position_size := v_position_size * 0.50;
    END IF;

    RETURN QUERY SELECT v_position_size, v_sl_distance, (p_tp_ratio * v_sl_distance), v_used_atr;
END;
$$ LANGUAGE plpgsql;

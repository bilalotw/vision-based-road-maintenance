def maintenance_priority(severity, extent_0_1, consequence_0_1, uncertainty,
                         severity_weights=None, review_threshold=0.35):
    # Prototype decision aid only; weights require engineering validation.
    weights=severity_weights or {0:0.25,1:0.60,2:1.0}
    if severity not in weights: raise ValueError("severity must map to configured ordinal weights")
    extent=max(0.0,min(1.0,float(extent_0_1)))
    consequence=max(0.0,min(1.0,float(consequence_0_1)))
    score=weights[severity]*extent*consequence
    priority="urgent" if score>=0.65 else "high" if score>=0.35 else "moderate" if score>=0.15 else "routine"
    return {"score":score,"priority":priority,
            "inspection_action":"manual_review" if uncertainty>=review_threshold else "automated_assessment"}

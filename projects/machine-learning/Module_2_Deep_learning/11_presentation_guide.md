# Final Presentation: Slide-by-Slide Guide

## Presentation Format

- **Duration:** 15 minutes presenting + 10 minutes Q&A
- **Audience:** CIO (business focus), Head of Risk (risk focus), Technical Committee (methodology focus)
- **Format:** Slide deck (PowerPoint, Google Slides, or PDF) + optional live demo
- **Tip:** Practice at least twice. Cut anything over 15 minutes. It is better to finish early and take more questions than to rush through slides.

---

## Slide 1: Title Slide

**Content:**
- Title: 
- Your name
- Date

**Speaker Notes:**
Keep this slide up while people settle in. No need to narrate it.

---

## Slide 2: The Problem (1 min)

---

## Slide 3: Our Approach (2 min)

---

## Slide 4: Model Comparison Table (1 min)

---

## Slide 5: The Equity Curve (2 min) - THE MOST IMPORTANT SLIDE

---

## Slide 6: Monthly Returns (1 min)

---

## Slide 7: Risk Analysis (2 min)

---

## Slide 8: Transaction Cost Sensitivity (1 min)

---

## Slide 9: Limitations and Failure Modes (2 min)

---

## Slide 10: Recommendation (2 min)

---

## Slide 11: Appendix / Backup (do not present - only for Q&A)

---

## Q&A Preparation: Questions You Will Be Asked

Prepare answers for these questions before the presentation:

**From the CIO:**
- "If we deployed this today, how much would we make per year?"
- "How does this compare to what we pay for third-party signals?"
- "What happens if the market crashes next week?"
- "How quickly can you get this into production?"

**From the Head of Risk:**
- "What is the worst month in your backtest? What caused it?"
- "Is the model correlated with our existing strategies? If so, it does not diversify our risk."
- "How do you know the model will not blow up on out-of-distribution data?"
- "What circuit breakers would you put in place?"

**From the Technical Committee:**
- "How did you prevent look-ahead bias?"
- "Show me the walk-forward results. Is performance stable across folds?"
- "Why did you choose [model] over [other model]?"
- "What is the retraining cadence? How do you detect model degradation?"

#!/usr/bin/env python3
"""
OpacityLens - a lensing tool built from static opaque lenses.
Kibler AI Solutions Corp.

What it does, in one line:
  Noisy signals in (barcode scans, Doppler readings) -> clean classified
  output, after passing through a stack of fixed opaque lenses, with a
  receipt from EVERY lens so you can SEE what each one did.

Why it exists:
  This is the monocle. It is the read-side companion to OpacityChunk.
  Where the chunker gives every CHUNK a receipt, this gives every LENS a
  receipt for every signal that passes through it. Nothing in the optical
  path is a black box. If a signal got dimmed, you can see which lens dimmed
  it and by how much. If a signal got blocked, you can see which lens
  blocked it and why.

The physics is real, not decorative:
  Each lens has a fixed OPACITY in [0,1]. Its transmittance is (1 - opacity).
  When you stack opaque lenses, transmittance MULTIPLIES through the stack -
  that is how real stacked filters behave (Beer-Lambert intuition). A signal
  enters at full intensity (1.0) and is dimmed by every lens it survives.
  If cumulative intensity falls below the floor, the signal does not make it
  out the far side. So a weak signal can be killed by accumulated dimming
  alone, with no single lens "rejecting" it - exactly like looking through
  too many smoked-glass plates.

  STATIC means every lens fixes its parameters at construction. No lens
  adapts to the signal. That is deliberate: a static lens is auditable. You
  can read its opacity and its rule once and trust it for every signal,
  instead of chasing adaptive behavior you cannot reproduce.

This is the PUBLIC-FACING lens discipline. It is simple and inspectable on
purpose. It does not contain the internal allocation logic.
"""

import re
import json
import math
import hashlib


def _finite(x, default=None):
    """Coerce x to a finite float, or return default. The single guard that
    stops NaN/inf and unconvertible payloads from poisoning the stack."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# --------------------------------------------------------------------------
# Signals - what goes IN to the monocle
# --------------------------------------------------------------------------

@dataclass
class Signal:
    """One raw reading entering the lens stack.

    kind       : 'barcode' or 'doppler' - which sensor it came from
    payload    : the read itself. For barcode, the code string. For doppler,
                 a signed number (Hz shift or m/s; +approaching, -receding).
    confidence : the sensor's own raw 0..1 trust in this reading. This is the
                 starting INTENSITY before any lens dims it.
    t          : timestamp in seconds (float). Used by the debounce lens.
    fingerprint: short stable hash of (kind, payload). Lets duplicates be SEEN.
    """
    kind: str
    payload: object
    confidence: float
    t: float
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            raw = f"{self.kind}:{self.payload}".encode("utf-8")
            self.fingerprint = hashlib.sha256(raw).hexdigest()[:12]


# --------------------------------------------------------------------------
# Receipts - the truth each lens leaves behind
# --------------------------------------------------------------------------

@dataclass
class LensReceipt:
    """What ONE lens did to ONE signal. This is the audit line.

    Every field is something a human can read to trust the optical path
    instead of taking it on faith.
    """
    lens: str             # which lens this receipt is from
    opacity: float        # the lens's fixed opacity (how dark its glass is)
    passed: bool          # did the signal survive this lens at all?
    reason: str           # plain-language why - "in band", "duplicate within 0.30s", etc.
    intensity_in: float   # signal strength arriving at this lens
    intensity_out: float  # signal strength leaving this lens (0.0 if blocked)


@dataclass
class LensedSignal:
    """A signal after the full stack, with its complete receipt trail."""
    fingerprint: str
    kind: str
    payload: object
    final_state: str          # classification, or 'BLOCKED'
    survived: bool            # did it make it out the far side of the monocle?
    final_intensity: float    # how much got through, 0..1
    blocked_by: Optional[str] # name of the first lens that hard-blocked it, if any
    receipts: List[LensReceipt] = field(default_factory=list)


# --------------------------------------------------------------------------
# Lenses - the static opaque plates of the monocle
# --------------------------------------------------------------------------

class Lens:
    """Base lens. A lens does exactly two things, in this order:

      1. Optionally HARD-BLOCK the signal for a logical reason (returns a
         reason string; intensity goes to 0). This is the lens being opaque
         to a whole category of signal.
      2. If not blocked, ATTENUATE the signal: intensity *= (1 - opacity).
         This is the static dimming every signal pays just to pass through.

    Subclasses override `block_reason` and/or `classify`. The base lens is a
    pure neutral-density filter: it never logic-blocks, it only dims.
    """

    def __init__(self, name: str, opacity: float):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"{name}: opacity must be in [0,1], got {opacity}")
        self.name = name
        self.opacity = opacity
        self.transmittance = 1.0 - opacity

    def applies_to(self, sig: Signal) -> bool:
        """Whether this lens acts on this kind of signal. Default: all kinds."""
        return True

    def block_reason(self, sig: Signal) -> Optional[str]:
        """Return a reason string to HARD-BLOCK, or None to let it pass.
        Base lens never hard-blocks."""
        return None

    def classify(self, sig: Signal) -> Optional[str]:
        """Optional: a lens may attach a state label as it passes the signal.
        Base lens attaches nothing."""
        return None


class DebounceLens(Lens):
    """Opaque to chatter. Blocks a reading that is identical (same fingerprint)
    to one it already saw within `window` seconds.

    This is the 'debounce' in your architecture made literal: the same scan
    bouncing twice in a few milliseconds is one event, not two. Static: the
    window is fixed, and the lens remembers only the last-seen time per
    fingerprint - simple enough to audit.
    """

    def __init__(self, opacity: float = 0.0, window: float = 0.30):
        super().__init__("debounce", opacity)
        self.window = window
        self._last_seen = {}  # fingerprint -> last timestamp

    def block_reason(self, sig: Signal) -> Optional[str]:
        prev = self._last_seen.get(sig.fingerprint)
        self._last_seen[sig.fingerprint] = sig.t
        # Prune entries older than the window so this dict can't grow forever
        # on a daemon running for weeks. Bounded memory is the 'runs forever'
        # guarantee made true.
        if len(self._last_seen) > 4096:
            cutoff = sig.t - self.window
            self._last_seen = {k: v for k, v in self._last_seen.items() if v >= cutoff}
        if prev is not None and (sig.t - prev) < self.window:
            return f"duplicate within {self.window:.2f}s of prior read"
        return None


class BarcodeBandLens(Lens):
    """Opaque to malformed barcodes. Only acts on barcode signals.

    Static rule: the code must be all digits and of an allowed length. This
    is intentionally a dumb, readable check, not a full symbology validator -
    you can confirm the rule by eye. Swap in a real check-digit routine when
    a specific symbology (EAN-13, Code128) is committed to; flagged honestly.
    """

    def __init__(self, opacity: float = 0.10, lengths=(8, 12, 13)):
        super().__init__("barcode_band", opacity)
        self.lengths = set(lengths)

    def applies_to(self, sig: Signal) -> bool:
        return sig.kind == "barcode"

    def block_reason(self, sig: Signal) -> Optional[str]:
        code = str(sig.payload)
        if not code.isdigit():
            return "non-numeric barcode payload"
        if len(code) not in self.lengths:
            return f"length {len(code)} not in allowed {sorted(self.lengths)}"
        return None


class DopplerBandLens(Lens):
    """Opaque to out-of-range Doppler, and the lens that names the motion state.
    Only acts on doppler signals.

    Static bands (signed reading; +approaching, -receding):
        |v| <  static_band      -> 'STATIC'
        v   >= static_band       -> 'APPROACHING'
        v   <= -static_band      -> 'RECEDING'
    A reading whose magnitude exceeds `clip` is hard-blocked as implausible -
    that is the lens being opaque to sensor glitches.
    """

    def __init__(self, opacity: float = 0.05, static_band: float = 0.5, clip: float = 1000.0):
        super().__init__("doppler_band", opacity)
        self.static_band = static_band
        self.clip = clip

    def applies_to(self, sig: Signal) -> bool:
        return sig.kind == "doppler"

    def block_reason(self, sig: Signal) -> Optional[str]:
        v = _finite(sig.payload)
        if v is None:
            return "doppler payload not a finite number"
        if abs(v) > self.clip:
            return f"magnitude {abs(v):.1f} exceeds plausible clip {self.clip:.0f}"
        return None

    def classify(self, sig: Signal) -> Optional[str]:
        v = _finite(sig.payload)
        if v is None:
            return None  # block_reason already caught it; never misclassify
        if abs(v) < self.static_band:
            return "STATIC"
        return "APPROACHING" if v > 0 else "RECEDING"


class CoherenceLens(Lens):
    """The smoked glass. Acts on every signal and dims hardest. Its opacity is
    high on purpose: this is the lens that makes weak signals fail by
    accumulated dimming rather than by any single rejection.

    It does not hard-block. It just dims. A low-confidence reading entering a
    high-opacity coherence lens, after already paying the earlier lenses, is
    the case that quietly falls below the floor - and the receipts show
    exactly that, so the death is visible, not silent.
    """

    def __init__(self, opacity: float = 0.30):
        super().__init__("coherence", opacity)


class EntropyLens(Lens):
    """Opaque to garbage. Hard-blocks a payload whose Shannon entropy per
    character is too high - i.e. it looks like random bytes, not a real code
    or reading. This is the Ollie Ledger entropy check made into glass.

    Static: the threshold (bits/char) is fixed. Numbers and short codes have
    low entropy; random base64-ish noise has high entropy. Doppler payloads
    are numeric and pass trivially. Wrapped in a guard so a weird payload can
    never raise - worst case it passes untouched, never crashes the stack.
    """

    def __init__(self, opacity: float = 0.0, max_bits_per_char: float = 4.2):
        super().__init__("entropy", opacity)
        self.max_bits = max_bits_per_char

    @staticmethod
    def _entropy(s: str) -> float:
        import math
        if not s:
            return 0.0
        counts = {}
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
        n = len(s)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def block_reason(self, sig: Signal) -> Optional[str]:
        try:
            s = str(sig.payload)
            if len(s) < 8:           # too short to judge; let it through
                return None
            bits = self._entropy(s)
            if bits > self.max_bits:
                return f"entropy {bits:.2f} bits/char exceeds {self.max_bits} - looks like noise"
        except Exception:
            return None              # never crash on a weird payload
        return None


class SecretsLens(Lens):
    """Opaque to credentials. Hard-blocks a payload that matches common
    secret shapes (API keys, tokens, private-key headers). This is the Ollie
    Ledger secrets detection as a lens.

    Static and DELIBERATELY NARROW: broad secret regexes are exactly what
    false-positived on ordinary operational language in the ledger. These
    patterns are tight on purpose. Honest note: this catches shaped secrets,
    not every secret - it is a guard, not a vault.
    """

    _PATTERNS = [
        (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key header"),
        (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "sk- style API key"),
        (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "GitHub token"),
        (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
        (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "Slack token"),
    ]

    def __init__(self, opacity: float = 0.0):
        super().__init__("secrets", opacity)

    def block_reason(self, sig: Signal) -> Optional[str]:
        try:
            s = str(sig.payload)
            for pat, label in self._PATTERNS:
                if pat.search(s):
                    return f"payload matches {label} - refusing to pass a credential"
        except Exception:
            return None
        return None


class RateFloodLens(Lens):
    """Backpressure as glass. Does NOT hard-block; it DIMS harder the faster
    signals arrive. Tracks a rolling count in a fixed window; when the rate
    exceeds a soft limit, effective opacity rises so floods fade out by
    accumulated dimming rather than being cut. Load control, made visible.

    Static: window, soft_limit, and the max extra opacity are fixed. The
    extra dimming is applied at look() time, so the receipt shows the real
    opacity used under load.
    """

    def __init__(self, base_opacity: float = 0.05, window: float = 1.0,
                 soft_limit: int = 20, max_extra_opacity: float = 0.6):
        super().__init__("rate_flood", base_opacity)
        self.window = window
        self.soft_limit = soft_limit
        self.max_extra = max_extra_opacity
        self._stamps = []  # recent timestamps within the window

    def _effective_opacity(self, sig: Signal) -> float:
        # prune stamps outside the window, then add this one
        t = _finite(sig.t, default=0.0)
        cutoff = t - self.window
        self._stamps = [s for s in self._stamps if s >= cutoff]
        self._stamps.append(t)
        rate = len(self._stamps)
        if rate <= self.soft_limit:
            return self.opacity
        # scale extra opacity with how far over the limit we are, capped
        over = min(1.0, (rate - self.soft_limit) / max(self.soft_limit, 1))
        return min(1.0, self.opacity + over * self.max_extra)

    def applies_to(self, sig: Signal) -> bool:
        # update rate state here so it tracks ALL traffic, then dim via opacity
        self._eff = self._effective_opacity(sig)
        return True

    @property
    def transmittance(self) -> float:
        # dynamic: reflects current load. Falls back to base when unset.
        return 1.0 - getattr(self, "_eff", self.opacity)

    @transmittance.setter
    def transmittance(self, _value):
        pass  # base __init__ sets this; we compute it dynamically instead


class DriftLens(Lens):
    """Opaque to drift. DIMS (does not block) a numeric signal that has
    wandered far from its running baseline. Only acts on numeric payloads
    (doppler, or numeric barcodes if you want); the further from baseline,
    the darker the glass, capped.

    Static: the baseline is a simple exponential moving average with a fixed
    smoothing factor, and the dimming scale is fixed. A first reading defines
    the baseline and passes clean. Non-numeric payloads are skipped safely.
    """

    def __init__(self, base_opacity: float = 0.0, kinds=("doppler",),
                 alpha: float = 0.2, tolerance: float = 5.0,
                 max_extra_opacity: float = 0.7):
        super().__init__("drift", base_opacity)
        self.kinds = set(kinds)
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_extra = max_extra_opacity
        self._baseline = {}  # kind -> EMA value

    def applies_to(self, sig: Signal) -> bool:
        if sig.kind not in self.kinds:
            self._eff = self.opacity
            return False
        v = _finite(sig.payload)
        if v is None:
            self._eff = self.opacity
            return False
        base = self._baseline.get(sig.kind)
        if base is None:
            self._baseline[sig.kind] = v
            self._eff = self.opacity            # first reading sets baseline
            return True
        dev = abs(v - base) / max(self.tolerance, 1e-6)
        extra = min(1.0, dev) * self.max_extra
        self._eff = min(1.0, self.opacity + extra)
        # update baseline AFTER judging this reading
        self._baseline[sig.kind] = self.alpha * v + (1 - self.alpha) * base
        return True

    @property
    def transmittance(self) -> float:
        return 1.0 - getattr(self, "_eff", self.opacity)

    @transmittance.setter
    def transmittance(self, _value):
        pass


class ChangeResponseLens(Lens):
    """Responsiveness, earned by test. PASSES MORE LIGHT (lowers opacity) when
    the signal just changed - the moment a real event happens. Replaces the
    decorative golden-spiral mask, which testing showed did nothing.

    Empirical basis (spiral_test.py): weighting up the points where the signal
    jumped won 4 of 6 scenarios vs a plain EMA - it catches steps and tracks
    sines faster. Use this for signals that MOVE (motion, event sensors).

    Mechanism: tracks last value per kind; a large jump from it temporarily
    clears the glass (opacity drops toward 0) so the changed reading lands at
    full strength. Static params: the jump scale and how much it clears.
    """

    def __init__(self, base_opacity: float = 0.2, kinds=("doppler",),
                 jump_scale: float = 5.0, max_clear: float = 0.2):
        super().__init__("change_response", base_opacity)
        self.kinds = set(kinds)
        self.jump_scale = jump_scale
        self.max_clear = max_clear  # lowest opacity floor when fully cleared
        self._last = {}

    def applies_to(self, sig: Signal) -> bool:
        if sig.kind not in self.kinds:
            self._eff = self.opacity
            return False
        v = _finite(sig.payload)
        if v is None:
            self._eff = self.opacity
            return False
        prev = self._last.get(sig.kind)
        self._last[sig.kind] = v
        if prev is None:
            self._eff = self.opacity
            return True
        jump = min(1.0, abs(v - prev) / max(self.jump_scale, 1e-6))
        # bigger jump -> opacity drops from base toward max_clear (more light)
        self._eff = self.opacity - jump * (self.opacity - self.max_clear)
        return True

    @property
    def transmittance(self) -> float:
        return 1.0 - getattr(self, "_eff", self.opacity)

    @transmittance.setter
    def transmittance(self, _value):
        pass


class DeviationDampLens(Lens):
    """Outlier damping, earned by test. DIMS points far from the running mean.
    Replaces the spiral for signals that should be STEADY.

    Empirical basis (spiral_test.py): on steady/noisy-steady signals this beat
    a plain EMA by ~19% (0.350 vs 0.432 error) by weighting down noise spikes.
    Use this for baselines, identity reads, anything meant to hold constant.

    Mechanism: running mean per kind; the further a reading sits from it, the
    darker the glass, capped. Distinct from DriftLens - drift judges vs a slow
    EMA baseline for *alarming*; this judges vs the mean for *noise rejection*.
    """

    def __init__(self, base_opacity: float = 0.0, kinds=("doppler",),
                 alpha: float = 0.1, scale: float = 3.0,
                 max_extra_opacity: float = 0.6):
        super().__init__("deviation_damp", base_opacity)
        self.kinds = set(kinds)
        self.alpha = alpha
        self.scale = scale
        self.max_extra = max_extra_opacity
        self._mean = {}

    def applies_to(self, sig: Signal) -> bool:
        if sig.kind not in self.kinds:
            self._eff = self.opacity
            return False
        v = _finite(sig.payload)
        if v is None:
            self._eff = self.opacity
            return False
        mean = self._mean.get(sig.kind)
        if mean is None:
            self._mean[sig.kind] = v
            self._eff = self.opacity
            return True
        dev = min(1.0, abs(v - mean) / max(self.scale, 1e-6))
        self._eff = min(1.0, self.opacity + dev * self.max_extra)
        self._mean[sig.kind] = self.alpha * v + (1 - self.alpha) * mean
        return True

    @property
    def transmittance(self) -> float:
        return 1.0 - getattr(self, "_eff", self.opacity)

    @transmittance.setter
    def transmittance(self, _value):
        pass


# --------------------------------------------------------------------------
# The monocle - an ordered stack of lenses
# --------------------------------------------------------------------------

class LensStack:
    """The monocle. An ordered list of static opaque lenses plus a floor.

    A signal travels front-to-back. At each applicable lens:
      - if the lens hard-blocks, intensity -> 0, we record the receipt, and
        the signal stops (later lenses still emit a 'not reached' receipt so
        the trail is complete and honest).
      - otherwise intensity *= transmittance, and any classification the lens
        offers becomes the running state.
    After the stack, if final intensity < floor, the signal is BLOCKED by
    accumulated opacity even if no single lens rejected it.
    """

    def __init__(self, lenses: List[Lens], floor: float = 0.15):
        self.lenses = lenses
        self.floor = floor

    def look(self, sig: Signal) -> LensedSignal:
        conf = _finite(sig.confidence, default=0.0)
        intensity = max(0.0, min(1.0, conf))
        state = "UNCLASSIFIED"
        blocked_by = None
        receipts: List[LensReceipt] = []

        for lens in self.lenses:
            if not lens.applies_to(sig):
                receipts.append(LensReceipt(
                    lens=lens.name, opacity=lens.opacity, passed=True,
                    reason="lens does not act on this signal kind",
                    intensity_in=round(intensity, 4), intensity_out=round(intensity, 4),
                ))
                continue

            if blocked_by is not None:
                receipts.append(LensReceipt(
                    lens=lens.name, opacity=lens.opacity, passed=False,
                    reason="not reached - signal already blocked upstream",
                    intensity_in=0.0, intensity_out=0.0,
                ))
                continue

            reason = lens.block_reason(sig)
            if reason is not None:
                blocked_by = lens.name
                receipts.append(LensReceipt(
                    lens=lens.name, opacity=lens.opacity, passed=False,
                    reason=reason,
                    intensity_in=round(intensity, 4), intensity_out=0.0,
                ))
                intensity = 0.0
                continue

            intensity_in = intensity
            intensity *= lens.transmittance
            label = lens.classify(sig)
            if label:
                state = label
            receipts.append(LensReceipt(
                lens=lens.name, opacity=lens.opacity, passed=True,
                reason=(f"classified {label}" if label else f"attenuated x{lens.transmittance:.2f}"),
                intensity_in=round(intensity_in, 4), intensity_out=round(intensity, 4),
            ))

        survived = blocked_by is None and intensity >= self.floor
        if blocked_by is not None:
            final_state = "BLOCKED"
        elif not survived:
            final_state = "BLOCKED"
            blocked_by = f"floor ({self.floor}) - accumulated opacity"
        else:
            final_state = state

        return LensedSignal(
            fingerprint=sig.fingerprint, kind=sig.kind, payload=sig.payload,
            final_state=final_state, survived=survived,
            final_intensity=round(intensity, 4), blocked_by=blocked_by,
            receipts=receipts,
        )


# --------------------------------------------------------------------------
# Health report - the RATML-family read on a whole pass
# --------------------------------------------------------------------------

def health_report(results: List[LensedSignal]) -> dict:
    """Plain-language health on a batch of looked-at signals.

    Same opacity-lens philosophy as OpacityChunk: SEE the quality, do not
    trust it blindly. Surfaced as the RATML-family numbers:

      recoverability - what fraction survived the stack (clean signal through)
      coherence      - consistency of surviving intensity (tight vs scattered)
      load           - how many signals, how many of each kind
      density        - average dimming paid by survivors (how dark the glass)
    """
    if not results:
        return {"error": "no signals processed - was the input empty?"}

    total = len(results)
    survived = [r for r in results if r.survived]
    n_surv = len(survived)
    fingerprints = [r.fingerprint for r in results]
    duplicates = len(fingerprints) - len(set(fingerprints))

    recoverability = round(100 * n_surv / total, 1)

    if survived:
        intens = [r.final_intensity for r in survived]
        avg_int = sum(intens) / n_surv
        spread = round((max(intens) - min(intens)) / max(avg_int, 1e-6), 2)
        avg_dimming = round(1.0 - avg_int, 3)
    else:
        avg_int, spread, avg_dimming = 0.0, 0.0, 1.0

    kinds = {}
    for r in results:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1

    return {
        "total_signals": total,
        "survived": n_surv,
        "blocked": total - n_surv,
        "recoverability_pct": recoverability,      # higher = cleaner path
        "duplicate_signals": duplicates,           # silent failure made visible
        "load_by_kind": kinds,
        "avg_survivor_intensity": round(avg_int, 3),
        "intensity_spread": spread,                # lower = more consistent
        "avg_dimming_paid": avg_dimming,           # how dark the stack runs
        "verdict": (
            "CLEAN - signals through the monocle are trustworthy"
            if recoverability >= 50 and duplicates == 0
            else "REVIEW - heavy blocking or duplicates need a look"
        ),
    }


def look_through(signals: List[Signal], stack: LensStack) -> dict:
    """The one call most people will use: signals + a monocle in, results + report out."""
    results = [stack.look(s) for s in signals]
    return {
        "report": health_report(results),
        "signals": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------
# Default monocle + demo
# --------------------------------------------------------------------------

def default_monocle() -> LensStack:
    """A sensible starter stack, front to back:
        debounce        - kill chatter first, cheaply
        secrets         - refuse credential-shaped payloads (hard block)
        entropy         - refuse random-noise payloads (hard block)
        barcode_band    - reject malformed codes (barcode only)
        doppler_band    - reject glitches + name the motion state (doppler only)
        drift           - dim numeric signals wandering off baseline
        rate_flood      - dim harder under signal floods (backpressure)
        coherence       - the smoked glass that weak signals die against

    Order is intentional: cheap hard-blocks (secrets, entropy) come early so
    bad input dies before paying for the rest; the dimming lenses (drift,
    rate_flood, coherence) come last so a signal's final intensity reflects
    the full cumulative opacity it survived.
    """
    return LensStack([
        DebounceLens(opacity=0.0, window=0.30),
        SecretsLens(opacity=0.0),
        EntropyLens(opacity=0.0, max_bits_per_char=4.2),
        BarcodeBandLens(opacity=0.10, lengths=(8, 12, 13)),
        DopplerBandLens(opacity=0.05, static_band=0.5, clip=1000.0),
        DriftLens(base_opacity=0.0, kinds=("doppler",), tolerance=5.0),
        RateFloodLens(base_opacity=0.05, window=1.0, soft_limit=20),
        CoherenceLens(opacity=0.30),
    ], floor=0.15)


if __name__ == "__main__":
    monocle = default_monocle()
    demo = [
        Signal("barcode", "036000291452", confidence=0.95, t=0.00),  # clean EAN-12-ish
        Signal("barcode", "036000291452", confidence=0.95, t=0.10),  # duplicate -> debounce
        Signal("barcode", "12ab", confidence=0.90, t=1.00),          # malformed -> band block
        Signal("barcode", "036000291452", confidence=0.22, t=2.00),  # weak -> dies on floor
        Signal("doppler", 14.2, confidence=0.88, t=3.00),            # approaching
        Signal("doppler", -9.0, confidence=0.80, t=4.00),            # receding
        Signal("doppler", 0.1, confidence=0.70, t=5.00),             # static
        Signal("doppler", 5000.0, confidence=0.99, t=6.00),          # glitch -> clip block
    ]
    print(json.dumps(look_through(demo, monocle), indent=2))

package expo.modules.krsemulator

import android.graphics.Rect
import android.graphics.RectF
import org.json.JSONObject
import kotlin.math.min
import kotlin.math.roundToInt

// In Focus, the small screen's height as a share of the large one's.
private const val FOCUS_SMALL_SHARE = 0.4f
// In Pop-up, the popped-up screen's height as a share of the one it covers.
private const val POPUP_SHARE = 0.78f
private const val MAX_GAP_DP = 64f

/** How the DS's two screens are arranged. Keys match ScreenMode in lib/screenLayout.ts. */
internal enum class ScreenMode(val key: String) {
  STACKED("stacked"),
  SIDE_BY_SIDE("sideBySide"),
  FOCUS("focus"),
  SINGLE("single"),
  /** One screen like the GBA, with the other popping up over it on demand. */
  POPUP("popup");

  companion object {
    fun fromKey(key: String?): ScreenMode = entries.firstOrNull { it.key == key } ?: STACKED
  }
}

/**
 * The player's choice of DS screen arrangement. [swapped] puts the touchscreen first: on top when
 * stacked, on the left side by side, large in Focus, and the one shown in Single and Pop-up (where
 * the top screen is then the one that pops up).
 */
internal data class DsScreenLayout(
  val mode: ScreenMode = ScreenMode.STACKED,
  val swapped: Boolean = false,
  val gapDp: Float = 0f,
) {
  companion object {
    fun parse(json: String): DsScreenLayout {
      if (json.isEmpty()) return DsScreenLayout()
      val parsed = JSONObject(json)
      return DsScreenLayout(
        mode = ScreenMode.fromKey(parsed.optString("mode")),
        swapped = parsed.optBoolean("swapped", false),
        gapDp = parsed.optDouble("gap", 0.0).toFloat().coerceIn(0f, MAX_GAP_DP),
      )
    }
  }
}

/**
 * Where the main (top) screen and the touchscreen go. An empty rectangle is a hidden screen. In
 * Pop-up, [popup] is the screen showing over [behindPopup]; both are empty otherwise.
 */
internal data class DsScreens(
  val main: Rect,
  val touch: Rect,
  val popup: Rect = Rect(),
  val behindPopup: Rect = Rect(),
) {
  /** The touchscreen is drawn first when the top screen pops up over it. */
  val touchFirst: Boolean
    get() = !popup.isEmpty && popup == main

  val bounds: Rect
    get() = Rect(main).apply { if (isEmpty) set(touch) else if (!touch.isEmpty) union(touch) }
}

private fun rect(left: Float, top: Float, width: Float, height: Float) =
  Rect(left.roundToInt(), top.roundToInt(), (left + width).roundToInt(), (top + height).roundToInt())

/** Width over height of each of a system's two screens. */
internal data class ScreenAspects(val top: Float, val touch: Float) {
  companion object {
    // Each DS screen is 256x192.
    val DS = ScreenAspects(256f / 192f, 256f / 192f)
    // The 3DS's top screen is 400x240 and its touchscreen 320x240.
    val N3DS = ScreenAspects(400f / 240f, 320f / 240f)

    fun of(system: GameSystem) = if (system == GameSystem.N3DS) N3DS else DS
  }
}

/** Fits the two screens into [box], in pixels, for the layout. [popupShown] is for Pop-up. */
internal fun arrangeDsScreens(
  box: RectF,
  layout: DsScreenLayout,
  density: Float,
  popupShown: Boolean = false,
  aspects: ScreenAspects = ScreenAspects.DS,
): DsScreens {
  val boxWidth = box.width()
  val boxHeight = box.height()
  val gap =
    if (layout.mode == ScreenMode.SINGLE || layout.mode == ScreenMode.POPUP) 0f else layout.gapDp * density
  // The first screen is the top one unless swapped. Screens line up by height, like on the device.
  val firstAspect = if (layout.swapped) aspects.touch else aspects.top
  val secondAspect = if (layout.swapped) aspects.top else aspects.touch
  val first: Rect
  val second: Rect

  when (layout.mode) {
    ScreenMode.STACKED -> {
      val height = min((boxHeight - gap) / 2, boxWidth / maxOf(firstAspect, secondAspect))
      val top = box.top + (boxHeight - height * 2 - gap) / 2
      val firstWidth = height * firstAspect
      val secondWidth = height * secondAspect
      first = rect(box.left + (boxWidth - firstWidth) / 2, top, firstWidth, height)
      second = rect(box.left + (boxWidth - secondWidth) / 2, top + height + gap, secondWidth, height)
    }
    ScreenMode.SIDE_BY_SIDE -> {
      val height = min(boxHeight, (boxWidth - gap) / (firstAspect + secondAspect))
      val left = box.left + (boxWidth - height * (firstAspect + secondAspect) - gap) / 2
      val top = box.top + (boxHeight - height) / 2
      first = rect(left, top, height * firstAspect, height)
      second = rect(left + height * firstAspect + gap, top, height * secondAspect, height)
    }
    ScreenMode.FOCUS -> {
      val share = FOCUS_SMALL_SHARE
      // The small screen goes beside the large one or under it, whichever leaves the large bigger.
      val besideHeight = min((boxWidth - gap) / (firstAspect + share * secondAspect), boxHeight)
      val underHeight = min(boxWidth / firstAspect, (boxHeight - gap) / (1 + share))
      if (besideHeight >= underHeight) {
        val height = besideHeight
        val width = height * firstAspect
        val top = box.top + (boxHeight - height) / 2
        // The large screen keeps the left edge and the small one grows into everything left, up to
        // the right edge, though never past the large one's size.
        val smallHeight = min((boxWidth - width - gap) / secondAspect, height)
        val smallWidth = smallHeight * secondAspect
        first = rect(box.left, top, width, height)
        // Top-aligned: in landscape the face buttons cover the bottom corner.
        second = rect(box.right - smallWidth, top, smallWidth, smallHeight)
      } else {
        val height = underHeight
        val width = height * firstAspect
        val smallHeight = height * share
        val smallWidth = smallHeight * secondAspect
        val top = box.top + (boxHeight - height - smallHeight - gap) / 2
        first = rect(box.left + (boxWidth - width) / 2, top, width, height)
        second = rect(box.left + (boxWidth - smallWidth) / 2, top + height + gap, smallWidth, smallHeight)
      }
    }
    ScreenMode.SINGLE, ScreenMode.POPUP -> {
      val height = min(boxHeight, boxWidth / firstAspect)
      val width = height * firstAspect
      first = rect(box.left + (boxWidth - width) / 2, box.top + (boxHeight - height) / 2, width, height)
      second = if (layout.mode == ScreenMode.POPUP && popupShown) {
        // Centered over the first, so it reads as a panel opened on top of the game.
        val popupHeight = height * POPUP_SHARE
        val popupWidth = min(popupHeight * secondAspect, width)
        rect(
          first.exactCenterX() - popupWidth / 2,
          first.exactCenterY() - popupHeight / 2,
          popupWidth,
          popupHeight,
        )
      } else {
        Rect()
      }
    }
  }
  if (layout.mode == ScreenMode.POPUP && !second.isEmpty) {
    return if (layout.swapped) {
      DsScreens(main = second, touch = first, popup = second, behindPopup = first)
    } else {
      DsScreens(main = first, touch = second, popup = second, behindPopup = first)
    }
  }
  return if (layout.swapped) DsScreens(main = second, touch = first) else DsScreens(main = first, touch = second)
}

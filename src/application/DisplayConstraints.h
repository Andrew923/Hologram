#pragma once

// Derived from the slicer/panel geometry:
// panel sweep is offset from spin axis, leaving an unswept center core.
static constexpr float CORE_RADIUS_PX = 12.0f;
static constexpr float CORE_MARGIN_PX = 2.0f;
static constexpr float CORE_SAFE_RADIUS_PX = CORE_RADIUS_PX + CORE_MARGIN_PX;

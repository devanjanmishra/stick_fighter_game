# Stick Fighter — Status Summary

## What's DONE

### Core Engine (10 Milestones — All Complete)
| # | Milestone | Module | Tests |
|---|-----------|--------|-------|
| M1 | Pose Estimation (MediaPipe) | `core/pose_estimator.py` | 8 |
| M2 | Front→Side-View Transform | `core/coordinate_transformer.py` | 10 |
| M3 | Keypoint Smoothing (One Euro Filter) | `core/smoothing.py` | 8 |
| M4 | Move Detection (Jab/Cross/Hook/Uppercut) | `core/move_detector.py` | 10 |
| M5 | Walking via Shoulder Tracking | `core/movement_tracker.py` | 8 |
| M6 | Calibration (DTW Template Matching) | `core/calibration.py` | 9 |
| M7 | NPC AI + Hitbox Collision | `game/npc.py`, `game/collision.py` | 10 |
| M8 | Combat System (HP, Rounds, Timer) | `game/combat_system.py`, `rendering/combat_ui.py` | 10 |
| M9 | Effects & Combos | `game/effects.py`, `game/combo_tracker.py`, `game/sound_manager.py` | 10 |
| M10 | NPC Styles + Difficulty | `game/npc_styles.py` | 10 |
| **Total** | | **16 modules** | **83 tests, all passing** |

### Video Validation (Your Camera Input)
- Processed your 15.4s video (462 frames, 1280x720, 30fps)
- **100% pose detection rate** — every frame had a valid pose
- **91 FPS** effective processing speed (11ms/frame on CPU)
- **77 moves detected**: 35 cross, 19 jab, 13 uppercut, 10 hook
- All 10 milestones pass end-to-end with real camera input
- Calibration templates generated from YOUR personal movements

### Bug Fix Applied This Session
- **Arms too long**: `z_to_x_scale` reduced from 500→150, added per-segment length clamping (40px upper arm + 40px forearm = 80px max, proportional to 70px torso). Arms now look anatomically correct.

### Documentation
- `README.md` — Full project overview, moves, calibration, NPC styles, roadmap
- `ARCHITECTURE.md` — Module API reference, data flow, algorithms, tuning constants, agent resumption guide
- `rendering/move_explainer.py` — In-app tutorial/calibration screens

---

## What's LEFT

### Immediate Next Steps (to make it playable)
| Priority | Task | Effort | Description |
|----------|------|--------|-------------|
| **P0** | **Wire up real-time game loop** | 1-2 days | Connect all modules: camera → pose → transform → smooth → detect → combat → render. Single `main.py` with Pygame event loop. |
| **P1** | **Tune move detection** | 1 day | Reduce cross over-detection (35 vs expected ~15-20). Add shoulder rotation check for jab vs cross discrimination. Increase cooldown from 8→12 frames. |
| **P1** | **NPC stick figure renderer** | 0.5 day | Currently NPC is a simple red circle+line. Need to render it as a proper stick figure with arm animations for attacks. |
| **P2** | **Sound effects integration** | 0.5 day | `SoundManager` exists but needs to be wired into the game loop. Hit/whoosh/KO sounds. |
| **P2** | **Calibration UI flow** | 1 day | Wire `MoveExplainer` screens into the game startup — user records moves before playing. |

### Medium-Term (Full Game Feel)
| Task | Effort | Description |
|------|--------|-------------|
| Blocking/defense mechanic | 1-2 days | Player can guard by crossing arms. NPC already has `block_chance`. |
| Leg tracking | 2-3 days | Add knee/ankle from MediaPipe. Enable kicks (front kick, roundhouse). |
| Round transitions & menus | 1-2 days | Start screen, character select, win/lose screens, replay. |
| Combo display & announcer | 1 day | On-screen combo counter, "DOUBLE!", "TRIPLE!" announcements. |
| Background & stage art | 1-2 days | Fighting arena backgrounds, parallax scrolling. |

### Long-Term (Production / Mobile)
| Task | Effort | Description |
|------|--------|-------------|
| RL-trained NPC opponent | 2-4 weeks | PPO self-play training in simulation. Export tiny model for inference. |
| Android port (Kivy/Buildozer) | 1-2 weeks | Package as APK. Use phone camera. |
| TV/PC streaming mode | 1-2 weeks | Phone camera streams to TV/PC display via WebRTC or local WiFi. |
| Play Store submission | 1 week | Developer account, store listing, APK signing, review. |
| Multiplayer (local/online) | 3-4 weeks | Two cameras, or networked play via WebSocket. |

### Technical Debt
- Move `import math` out of `transform()` method to module level (minor)
- Add integration test that runs the full pipeline end-to-end
- Add type hints to validation script
- Consider replacing rule-based detection with DTW template matching as primary (calibration data is already captured)

"""Apex Predator — Tactical, Map-Aware Battle Royale Bot for mechfight/1."""
from __future__ import annotations

from array import array
import heapq
import json
import math
import os
import sys
import time
import mechfight as mf


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def point_in_poly(x: float, y: float, ring: list[float]) -> bool:
    inside = False
    n = len(ring) // 2
    if n < 3:
        return False
    x0, y0 = ring[2 * (n - 1)], ring[2 * (n - 1) + 1]
    for i in range(n):
        x1, y1 = ring[2 * i], ring[2 * i + 1]
        if ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1):
            inside = not inside
        x0, y0 = x1, y1
    return inside


def seg_seg_intersect_t(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> float | None:
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-9:
        return None
    qpx, qpy = cx - ax, cy - ay
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


class ArenaMap:
    def __init__(self, map_dir: str, step_up: float = 1.2, mech_height: float = 9.0):
        world_path = os.path.join(map_dir, "v2", "world.json")
        terr_path = os.path.join(map_dir, "v2", "terrain.i16")

        with open(world_path, "r", encoding="utf-8") as f:
            world = json.load(f)

        tinfo = world["terrain"]
        self.t_min_x = float(tinfo["minX"])
        self.t_min_y = float(tinfo["minY"])
        self.t_step = float(tinfo["step"])
        self.t_nx = int(tinfo["nx"])
        self.t_ny = int(tinfo["ny"])
        self.t_scale = float(tinfo["scale"])

        arr = array("h")
        with open(terr_path, "rb") as f:
            arr.frombytes(f.read())
        if sys.byteorder != "little":
            arr.byteswap()
        self.terrain = arr

        b = world["bounds"]
        self.min_x = float(b["minX"])
        self.max_x = float(b["maxX"])
        self.min_y = float(b["minY"])
        self.max_y = float(b["maxY"])

        # Preprocess prisms with AABBs
        self.prisms = []
        for p in world["prisms"]:
            ring = p["ring"]
            xs = ring[0::2]
            ys = ring[1::2]
            if not xs:
                continue
            self.prisms.append({
                "kind": p["kind"],
                "base": float(p["base"]),
                "top": float(p["top"]),
                "minx": min(xs),
                "maxx": max(xs),
                "miny": min(ys),
                "maxy": max(ys),
                "ring": ring,
                "holes": p.get("holes") or [],
            })

        # Build 5m nav grid for fast A* pathfinding
        self.cell = 5.0
        self.gx0 = self.min_x
        self.gy0 = self.min_y
        self.gnx = int((self.max_x - self.min_x) / self.cell) + 1
        self.gny = int((self.max_y - self.min_y) / self.cell) + 1
        self.blocked = bytearray(self.gnx * self.gny)

        pad = 2.6
        for pr in self.prisms:
            ix0 = max(0, int((pr["minx"] - pad - self.gx0) / self.cell))
            ix1 = min(self.gnx - 1, int((pr["maxx"] + pad - self.gx0) / self.cell))
            iy0 = max(0, int((pr["miny"] - pad - self.gy0) / self.cell))
            iy1 = min(self.gny - 1, int((pr["maxy"] + pad - self.gy0) / self.cell))
            base, top = pr["base"], pr["top"]
            ring, holes = pr["ring"], pr["holes"]
            for iy in range(iy0, iy1 + 1):
                cy = self.gy0 + (iy + 0.5) * self.cell
                row_off = iy * self.gnx
                for ix in range(ix0, ix1 + 1):
                    if self.blocked[row_off + ix]:
                        continue
                    cx = self.gx0 + (ix + 0.5) * self.cell
                    gz = self.ground_z(cx, cy)
                    # Check if prism blocks walking at ground level (or clears head if bridge)
                    if base >= gz + mech_height or top <= gz + step_up:
                        continue
                    hit = False
                    for ox, oy in ((0.0, 0.0), (2.2, 0.0), (-2.2, 0.0), (0.0, 2.2), (0.0, -2.2)):
                        px, py = cx + ox, cy + oy
                        if point_in_poly(px, py, ring):
                            if not any(point_in_poly(px, py, h) for h in holes):
                                hit = True
                                break
                    if hit:
                        self.blocked[row_off + ix] = 1

        # Precompute cover penalty (cells adjacent to buildings get wall-clearance cost for smooth pathing,
        # while also letting us query cover spots)
        self.near_wall = bytearray(self.gnx * self.gny)
        for iy in range(1, self.gny - 1):
            ro = iy * self.gnx
            for ix in range(1, self.gnx - 1):
                if self.blocked[ro + ix]:
                    continue
                if (
                    self.blocked[ro + ix - 1]
                    or self.blocked[ro + ix + 1]
                    or self.blocked[ro - self.gnx + ix]
                    or self.blocked[ro + self.gnx + ix]
                ):
                    self.near_wall[ro + ix] = 1

    def ground_z(self, x: float, y: float) -> float:
        fx = clamp((x - self.t_min_x) / self.t_step, 0.0, self.t_nx - 1.001)
        fy = clamp((y - self.t_min_y) / self.t_step, 0.0, self.t_ny - 1.001)
        ix = int(fx)
        iy = int(fy)
        dx = fx - ix
        dy = fy - iy
        idx = iy * self.t_nx + ix
        t = self.terrain
        z00 = t[idx]
        z10 = t[idx + 1]
        z01 = t[idx + self.t_nx]
        z11 = t[idx + self.t_nx + 1]
        z = (
            z00 * (1.0 - dx) * (1.0 - dy)
            + z10 * dx * (1.0 - dy)
            + z01 * (1.0 - dx) * dy
            + z11 * dx * dy
        )
        return z * self.t_scale

    def point_inside_prism(self, x: float, y: float, z: float, pr: dict) -> bool:
        if z < pr["base"] or z > pr["top"]:
            return False
        if not point_in_poly(x, y, pr["ring"]):
            return False
        for h in pr["holes"]:
            if point_in_poly(x, y, h):
                return False
        return True

    def has_los(self, p1: list[float], p2: list[float]) -> bool:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        minx = x1 if x1 < x2 else x2
        maxx = x2 if x1 < x2 else x1
        miny = y1 if y1 < y2 else y2
        maxy = y2 if y1 < y2 else y1
        minz = z1 if z1 < z2 else z2
        maxz = z2 if z1 < z2 else z1

        # Terrain occlusion check (6 interior samples)
        for s in (0.15, 0.32, 0.5, 0.68, 0.85):
            tx = x1 + (x2 - x1) * s
            ty = y1 + (y2 - y1) * s
            tz = z1 + (z2 - z1) * s
            if tz < self.ground_z(tx, ty) + 0.15:
                return False

        # Prism occlusion check
        dz = z2 - z1
        for pr in self.prisms:
            if (
                pr["maxx"] < minx or pr["minx"] > maxx
                or pr["maxy"] < miny or pr["miny"] > maxy
                or pr["top"] < minz or pr["base"] > maxz
            ):
                continue
            if self.point_inside_prism(x1, y1, z1, pr) or self.point_inside_prism(x2, y2, z2, pr):
                return False
            ring = pr["ring"]
            n = len(ring) // 2
            px0, py0 = ring[2 * (n - 1)], ring[2 * (n - 1) + 1]
            for i in range(n):
                px1, py1 = ring[2 * i], ring[2 * i + 1]
                t = seg_seg_intersect_t(x1, y1, x2, y2, px0, py0, px1, py1)
                if t is not None:
                    iz = z1 + dz * t
                    if pr["base"] <= iz <= pr["top"]:
                        return False
                px0, py0 = px1, py1
        return True

    def to_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = int(clamp((x - self.gx0) / self.cell, 0, self.gnx - 1))
        iy = int(clamp((y - self.gy0) / self.cell, 0, self.gny - 1))
        return ix, iy

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return self.gx0 + (ix + 0.5) * self.cell, self.gy0 + (iy + 0.5) * self.cell

    def nearest_free(self, ix: int, iy: int) -> tuple[int, int]:
        if not self.blocked[iy * self.gnx + ix]:
            return ix, iy
        for r in range(1, 6):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.gnx and 0 <= ny < self.gny:
                        if not self.blocked[ny * self.gnx + nx]:
                            return nx, ny
        return ix, iy

    def find_next_waypoint(self, sx: float, sy: float, gx: float, gy: float) -> tuple[float, float]:
        six, siy = self.nearest_free(*self.to_grid(sx, sy))
        gix, giy = self.nearest_free(*self.to_grid(gx, gy))
        if (six, siy) == (gix, giy):
            return gx, gy

        start = siy * self.gnx + six
        goal = giy * self.gnx + gix
        gnx, gny = self.gnx, self.gny
        blocked = self.blocked
        near_wall = self.near_wall

        g_score = {start: 0.0}
        came_from: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(math.hypot(gix - six, giy - siy), start)]
        best_node = start
        best_h = math.hypot(gix - six, giy - siy)

        dirs = (
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        )
        expansions = 0
        while pq and expansions < 2000:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                best_node = cur
                break
            expansions += 1
            cy, cx = divmod(cur, gnx)
            cur_g = g_score[cur]
            for dx, dy, step_cost in dirs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < gnx and 0 <= ny < gny):
                    continue
                nid = ny * gnx + nx
                if blocked[nid]:
                    continue
                # Avoid cutting corners diagonally through blocked cells
                if dx != 0 and dy != 0:
                    if blocked[cy * gnx + nx] or blocked[ny * gnx + cx]:
                        continue
                penalty = 0.25 if near_wall[nid] else 0.0
                ng = cur_g + step_cost + penalty
                if ng < g_score.get(nid, 1e18):
                    g_score[nid] = ng
                    came_from[nid] = cur
                    h = math.hypot(gix - nx, giy - ny)
                    if h < best_h:
                        best_h = h
                        best_node = nid
                    heapq.heappush(pq, (ng + h * 1.15, nid))

        # Walk back to waypoint ~3 cells ahead for smooth steering
        chain = [best_node]
        cur = best_node
        while cur in came_from:
            cur = came_from[cur]
            chain.append(cur)
        chain.reverse()
        if len(chain) <= 1:
            return gx, gy
        pick_idx = min(3, len(chain) - 1)
        wy, wx = divmod(chain[pick_idx], gnx)
        return self.to_world(wx, wy)

    def find_cover_spot(self, sx: float, sy: float, sz: float, threat_pos: list[float],
                        zc: list[float], zr: float) -> tuple[float, float] | None:
        """Sample nearby walkable points to break LOS with threat while staying in zone."""
        tx, ty, tz = threat_pos
        threat_eye = [tx, ty, tz + 8.0]
        base_ang = math.atan2(sy - ty, sx - tx)
        best_pt = None
        best_score = -1e9
        for dist in (15.0, 28.0, 42.0):
            for offset in (0.0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8):
                ang = base_ang + offset
                cx = sx + math.cos(ang) * dist
                cy = sy + math.sin(ang) * dist
                if not (self.min_x + 5 <= cx <= self.max_x - 5 and self.min_y + 5 <= cy <= self.max_y - 5):
                    continue
                if math.hypot(cx - zc[0], cy - zc[1]) > zr - 12.0:
                    continue
                ix, iy = self.to_grid(cx, cy)
                if self.blocked[iy * self.gnx + ix]:
                    continue
                cz = self.ground_z(cx, cy)
                hidden = not self.has_los([cx, cy, cz + 4.5], threat_eye)
                if hidden:
                    sc = 100.0 - dist * 0.5 + (10.0 if self.near_wall[iy * self.gnx + ix] else 0.0)
                    if sc > best_score:
                        best_score = sc
                        best_pt = (cx, cy)
        return best_pt


UPGRADE_PRIORITY = {
    "armor": 100,
    "plating": 95,
    "nanites": 88,
    "servos": 82,
    "laser_damage": 78,
    "laser_charge": 76,
    "gun_rate": 74,
    "gun_damage": 72,
    "gun_velocity": 68,
    "lightning_range": 66,
    "lightning_efficiency": 64,
    "reactor": 60,
    "capacitor": 56,
    "jets": 52,
    "gun_mag": 50,
    "mortar_reload": 48,
    "mortar_velocity": 46,
    "mortar_splash": 44,
    "lightning_chain": 42,
    "laser_pierce": 40,
    "gyros": 36,
    "sensors": 30,
}

WEAPON_RANK = {
    "laser": 5,
    "lightning": 4,
    "gun": 3,
    "mortar": 1,
}


def solve_gun_lead(muzzle: list[float], target_pos: list[float], target_vel: list[float],
                   proj_speed: float, target_z_off: float = 4.5) -> tuple[float, float, float]:
    px = target_pos[0] - muzzle[0]
    py = target_pos[1] - muzzle[1]
    pz = (target_pos[2] + target_z_off) - muzzle[2]
    vx, vy, vz = target_vel

    a = vx * vx + vy * vy + vz * vz - proj_speed * proj_speed
    b = 2.0 * (px * vx + py * vy + pz * vz)
    c = px * px + py * py + pz * pz

    t = 0.0
    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            t = max(0.0, -c / b)
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sq = math.sqrt(disc)
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)
            cands = [x for x in (t1, t2) if x > 0.0]
            if cands:
                t = min(cands)
    t = min(t, 0.75)
    ax = px + vx * t
    ay = py + vy * t
    az = pz + vz * t
    horiz = math.hypot(ax, ay)
    yaw = math.atan2(ay, ax)
    pitch = math.atan2(az, max(horiz, 1e-6))
    return yaw, pitch, math.hypot(horiz, az)


def solve_mortar_pitch(dx: float, dz: float, speed: float, g: float, high_arc: bool = False) -> float | None:
    if dx < 1.0:
        return None
    v2 = speed * speed
    v4 = v2 * v2
    disc = v4 - g * (g * dx * dx + 2.0 * dz * v2)
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    tan_theta = (v2 + sq) / (g * dx) if high_arc else (v2 - sq) / (g * dx)
    return math.atan(tan_theta)


class Bot:
    name = "apex-predator"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.reach = float(cfg["loot"]["pickup_radius"])
        mech_cfg = cfg["mech"]
        map_dir = hello["map"]["dir"]
        if not os.path.isdir(map_dir) and os.path.isdir("/arena"):
            map_dir = "/arena"
        self.arena = ArenaMap(
            map_dir,
            step_up=float(mech_cfg.get("step_up", 1.2)),
            mech_height=float(mech_cfg.get("height", 9.0)),
        )
        self.min_pitch = float(mech_cfg.get("min_pitch", -0.7))
        self.max_pitch = float(mech_cfg.get("max_pitch", 1.3))

        self.last_pos: list[float] | None = None
        self.stuck_ticks = 0
        self.strafe_dir = 1
        self.strafe_timer = 0
        self.nav_wp: tuple[float, float] | None = None
        self.nav_tick = -100
        self.nav_goal: tuple[float, float] | None = None
        self.last_threat_bearing: float | None = None
        self.last_threat_t = -100.0
        self.recent_sounds: list[tuple[float, float, float]] = []  # (t, bearing, approx_dist)
        self.charging_laser = False

    def pick_upgrade(self, offer: dict) -> int:
        choices = offer["choices"]
        best_i = 0
        best_sc = -1
        for i, ch in enumerate(choices):
            sc = UPGRADE_PRIORITY.get(ch["id"], 10)
            if sc > best_sc:
                best_sc = sc
                best_i = i
        return best_i

    def maybe_loot_weapon(self, me: dict, pickups: list[dict]) -> dict:
        x, y, z = me["pos"]
        held = me["weapons"]
        held_kinds = [w["kind"] if w else None for w in held]
        for p in pickups:
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) <= self.reach - 0.15 and abs(p["pos"][2] - z) <= 3.8:
                kind = p["kind"]
                if kind in held_kinds:
                    continue
                if held[1] is None:
                    return {"take": p["id"]}
                if held[0] is None:
                    return {"take": p["id"]}
                # Swap worse weapon if not currently charging laser in that slot
                r_new = WEAPON_RANK.get(kind, 0)
                worst_slot = None
                worst_rank = r_new
                for s_idx, w in enumerate(held):
                    if w["kind"] == "laser" and w.get("charge", 0.0) > 0.0:
                        continue
                    r_w = WEAPON_RANK.get(w["kind"], 0)
                    if r_w < worst_rank:
                        worst_rank = r_w
                        worst_slot = s_idx
                if worst_slot is not None:
                    return {"take": p["id"], "drop": worst_slot}
        return {}

    def act(self, obs: mf.Observation) -> mf.Action:
        t = float(obs["t"])
        tick = int(obs["tick"])
        me = obs["self"]
        pos = me["pos"]
        x, y, z = pos
        hp = float(me["hp"])
        max_hp = float(me["max_hp"])
        energy = float(me["energy"])
        hp_frac = hp / max(max_hp, 1.0)
        stats = me["stats"]
        muzzle_z = z + float(stats.get("muzzle_height", 7.0))
        muzzle = [x, y, muzzle_z]

        act: mf.Action = {"tick": tick}
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"])

        # Track damage/sound bearings for rear-armor protection
        for ev in obs.get("events", []):
            if ev["type"] == "damaged" and ev.get("bearing") is not None:
                self.last_threat_bearing = float(ev["bearing"])
                self.last_threat_t = t
        for snd in obs.get("sounds", []):
            if snd["dist"] in ("near", "mid"):
                b = float(snd["bearing"])
                self.last_threat_bearing = b
                self.last_threat_t = t
                approx_d = 35.0 if snd["dist"] == "near" else 100.0
                self.recent_sounds.append((t, b, approx_d))
        self.recent_sounds = [s for s in self.recent_sounds if t - s[0] < 2.5][-12:]

        # Stuck detection (only when we actually wanted to move)
        if self.last_pos is not None:
            moved = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
            self._last_moved = moved
        else:
            self._last_moved = 1.0
        self.last_pos = [x, y, z]

        # Classify enemies & NPCs
        enemies = []
        npcs = []
        for m in obs.get("mechs", []):
            d = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
            if m["kind"] == "mech":
                enemies.append((d, m))
            else:
                npcs.append((d, m))
        enemies.sort(key=lambda item: item[0])
        npcs.sort(key=lambda item: item[0])

        # Check immediate weapon pickup at feet
        loot_cmd = self.maybe_loot_weapon(me, obs.get("pickups", []))
        if loot_cmd:
            act.update(loot_cmd)

        # Zone geometry
        zone = obs["zone"]
        zc = zone["center"]
        zr = float(zone["radius"])
        nzc = zone["next_center"]
        nzr = float(zone["next_radius"])
        dist_cur_zc = math.hypot(x - zc[0], y - zc[1])
        dist_next_zc = math.hypot(x - nzc[0], y - nzc[1])
        outside_cur = (not me["in_zone"]) or (dist_cur_zc > max(zr - 8.0, 2.0))
        time_to_shrink = float(zone["shrink_start"]) - t
        need_next_zone = dist_next_zc > max(nzr - 18.0, 4.0) and (time_to_shrink < 18.0 or zr < 180.0)

        # Decide movement target (Survival-first!)
        primary_enemy = enemies[0][1] if enemies else None
        primary_dist = enemies[0][0] if enemies else 9999.0

        # Safe position inside next circle (choose edge angle away from visible enemies & combat sounds)
        base_ang = math.atan2(y - nzc[1], x - nzc[0])
        safe_r = min(nzr * 0.68, max(0.0, nzr - 14.0))
        best_safe_ang = base_ang
        best_safe_sc = -1e9
        for da in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):
            ang = base_ang + da
            px = nzc[0] + math.cos(ang) * safe_r
            py = nzc[1] + math.sin(ang) * safe_r
            sc = -abs(da) * 15.0
            for _, m in enemies:
                sc += min(math.hypot(px - m["pos"][0], py - m["pos"][1]), 160.0)
            for _, sb, sd in self.recent_sounds:
                sx_s = x + math.cos(sb) * sd
                sy_s = y + math.sin(sb) * sd
                sc += min(math.hypot(px - sx_s, py - sy_s), 140.0) * 0.4
            if sc > best_safe_sc:
                best_safe_sc = sc
                best_safe_ang = ang
        safe_pt = (nzc[0] + math.cos(best_safe_ang) * safe_r, nzc[1] + math.sin(best_safe_ang) * safe_r)

        goal_pt = safe_pt
        want_boost = False
        want_cover = (
            primary_enemy is not None
            and (hp_frac < 0.45 or len(enemies) >= 2 or (primary_enemy.get("charging") and primary_dist < 220.0))
        )

        if outside_cur:
            goal_pt = (nzc[0], nzc[1])
            want_boost = energy > 20.0
        elif need_next_zone:
            goal_pt = safe_pt
            want_boost = dist_next_zc > nzr and energy > 35.0
        elif want_cover and primary_enemy is not None:
            cov = self.arena.find_cover_spot(x, y, z, primary_enemy["pos"], nzc, max(nzr, 20.0))
            if cov is not None:
                goal_pt = cov
                want_boost = energy > 25.0 and hp_frac < 0.38
            else:
                # Kite away from enemy toward safe zone center
                away_ang = math.atan2(y - primary_enemy["pos"][1], x - primary_enemy["pos"][0])
                goal_pt = (x + math.cos(away_ang) * 35.0, y + math.sin(away_ang) * 35.0)
                want_boost = energy > 45.0
        else:
            # Opportunistic loot: nearby cubes -> upgrades -> safe supply drops -> safe pickups -> crates
            best_loot = None
            best_loot_d = 9999.0
            for c in obs.get("cubes", []):
                dc = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                if dc < 50.0 and dc < best_loot_d and (not enemies or dc < primary_dist * 0.7):
                    best_loot_d = dc
                    best_loot = (c["pos"][0], c["pos"][1])
            if best_loot is not None:
                goal_pt = best_loot
                want_boost = best_loot_d < 20.0 and energy > 55.0
            else:
                # Check nearby weapon upgrade pickup
                held_kinds = {w["kind"] for w in me["weapons"] if w}
                worst_rank = min((WEAPON_RANK.get(k, 0) for k in held_kinds), default=0)
                if len(held_kinds) < 2 or worst_rank < 3:
                    for p in obs.get("pickups", []):
                        if p["kind"] in held_kinds:
                            continue
                        if len(held_kinds) == 2 and WEAPON_RANK.get(p["kind"], 0) <= worst_rank:
                            continue
                        dp = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                        if dp < 55.0 and dp < best_loot_d and (not enemies or dp < 25.0):
                            best_loot_d = dp
                            best_loot = (p["pos"][0], p["pos"][1])
                if best_loot is not None:
                    goal_pt = best_loot
                else:
                    # Check uncontested supply drops inside zone
                    for dr in obs.get("drops", []):
                        dd = math.hypot(dr["pos"][0] - x, dr["pos"][1] - y)
                        if dd < 90.0 and dr["lands_in"] < 8.0 and (not enemies or primary_dist > 55.0):
                            goal_pt = (dr["pos"][0], dr["pos"][1])
                            want_boost = dd > 15.0 and energy > 50.0
                            break
                    else:
                        if primary_enemy is not None:
                            # Maintain optimal combat range with weapon-triangle counter spacing
                            e_weapons = set(primary_enemy.get("weapons", []))
                            held_k = {w["kind"] for w in me["weapons"] if w}
                            has_light = "lightning" in held_k
                            if has_light and ("laser" in e_weapons or hp_frac > 0.65):
                                ideal_d = 32.0
                            elif "lightning" in e_weapons and not has_light:
                                ideal_d = 78.0
                            else:
                                ideal_d = 68.0
                            self.strafe_timer += 1
                            if self.strafe_timer >= 14:
                                self.strafe_timer = 0
                                self.strafe_dir = -self.strafe_dir
                            to_e = math.atan2(primary_enemy["pos"][1] - y, primary_enemy["pos"][0] - x)
                            rad_push = 0.40 if primary_dist > ideal_d + 12.0 else (-0.50 if primary_dist < ideal_d - 10.0 else 0.0)
                            move_ang = to_e + self.strafe_dir * (math.pi * 0.5 - rad_push)
                            goal_pt = (x + math.cos(move_ang) * 20.0, y + math.sin(move_ang) * 20.0)
                        elif obs.get("crates") and hp_frac > 0.55:
                            cr = min(obs["crates"], key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
                            if math.hypot(cr["pos"][0] - x, cr["pos"][1] - y) < 105.0:
                                goal_pt = (cr["pos"][0], cr["pos"][1])
                        elif npcs and npcs[0][0] < 90.0 and hp_frac > 0.72:
                            goal_pt = (npcs[0][1]["pos"][0], npcs[0][1]["pos"][1])

        # Navigation waypoint via A* when far from goal
        dist_goal = math.hypot(goal_pt[0] - x, goal_pt[1] - y)
        if dist_goal > 22.0:
            recompute = (
                self.nav_wp is None
                or tick - self.nav_tick >= 8
                or self.nav_goal is None
                or math.hypot(goal_pt[0] - self.nav_goal[0], goal_pt[1] - self.nav_goal[1]) > 15.0
                or math.hypot(self.nav_wp[0] - x, self.nav_wp[1] - y) < 6.0
            )
            if recompute:
                self.nav_wp = self.arena.find_next_waypoint(x, y, goal_pt[0], goal_pt[1])
                self.nav_tick = tick
                self.nav_goal = goal_pt
            steer_pt = self.nav_wp
        else:
            steer_pt = goal_pt

        mx, my = steer_pt[0] - x, steer_pt[1] - y
        mlen = math.hypot(mx, my)
        if mlen > 0.5:
            mx /= mlen
            my /= mlen
        else:
            mx, my = 0.0, 0.0

        # Incoming laser-charge dodging (enemy aiming charging laser at us!)
        laser_dodged = False
        for d, m in enemies:
            if m.get("charging") and d < 275.0:
                to_us = math.atan2(y - m["pos"][1], x - m["pos"][0])
                aim_diff = abs(wrap_angle(float(m["aim"][0]) - to_us))
                if aim_diff < 0.28:
                    # Strafe sharply perpendicular to the enemy's aim line
                    perp_ang = float(m["aim"][0]) + self.strafe_dir * (math.pi * 0.5)
                    mx = 0.25 * mx + 0.95 * math.cos(perp_ang)
                    my = 0.25 * my + 0.95 * math.sin(perp_ang)
                    norm = math.hypot(mx, my)
                    if norm > 1e-6:
                        mx /= norm
                        my /= norm
                    if energy > 18.0:
                        want_boost = True
                    laser_dodged = True
                    break

        # Incoming projectile dodging (override move slightly sideways)
        if not laser_dodged:
            for pr in obs.get("projectiles", []):
                px, py, pz = pr["pos"]
                pvx, pvy, _ = pr["vel"]
                spd2 = pvx * pvx + pvy * pvy
                if spd2 < 1.0:
                    continue
                rx, ry = x - px, y - py
                t_closest = (rx * pvx + ry * pvy) / spd2
                if 0.0 < t_closest < 0.55:
                    cx = px + pvx * t_closest
                    cy = py + pvy * t_closest
                    miss_d = math.hypot(x - cx, y - cy)
                    if miss_d < 5.5:
                        # Evade perpendicular to projectile trajectory
                        pspd = math.sqrt(spd2)
                        nx, ny = -pvy / pspd, pvx / pspd
                        if nx * (x - cx) + ny * (y - cy) < 0:
                            nx, ny = -nx, -ny
                        mx = 0.35 * mx + 0.85 * nx
                        my = 0.35 * my + 0.85 * ny
                        norm = math.hypot(mx, my)
                        if norm > 1e-6:
                            mx /= norm
                            my /= norm
                        if energy > 25.0 and pr["kind"] == "mortar":
                            want_boost = True
                        break

        act["move"] = [mx, my]
        act["boost"] = bool(want_boost and (mlen > 0.5 or laser_dodged))
        if mlen > 0.5 and self._last_moved < 0.25:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 2)
        if self.stuck_ticks >= 5 and me["grounded"]:
            act["jump"] = True
            if self.stuck_ticks >= 9 and energy > 30.0:
                act["boost"] = True
        else:
            act["jump"] = False

        # Combat: choose target (enemy mech > aggressive npc > safe crate)
        fire = [False, False]
        aim_yaw, aim_pitch = me["aim"]

        target_obj = None
        target_is_crate = False
        if enemies:
            # Prefer low HP, triangle-counterable, or close visible enemy with LOS
            held_k = {w["kind"] for w in me["weapons"] if w}
            best_score = -1e9
            for d, m in enemies:
                los = self.arena.has_los(muzzle, [m["pos"][0], m["pos"][1], m["pos"][2] + 4.5])
                mhp = float(m.get("hp", 1000.0))
                ew = set(m.get("weapons", []))
                tri = 0.0
                if "laser" in held_k and "mortar" in ew:
                    tri += 180.0
                if "lightning" in held_k and m.get("charging") and d < 48.0:
                    tri += 220.0
                sc = (500.0 if los else 0.0) + tri - d - mhp * 0.15
                if sc > best_score:
                    best_score = sc
                    target_obj = m
        elif npcs and npcs[0][0] < 85.0 and hp_frac > 0.60:
            if self.arena.has_los(muzzle, [npcs[0][1]["pos"][0], npcs[0][1]["pos"][1], npcs[0][1]["pos"][2] + 2.0]):
                target_obj = npcs[0][1]
        elif obs.get("crates") and hp_frac > 0.55:
            cr = min(obs["crates"], key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
            if math.hypot(cr["pos"][0] - x, cr["pos"][1] - y) < 85.0:
                if self.arena.has_los(muzzle, [cr["pos"][0], cr["pos"][1], cr["pos"][2] + 1.5]):
                    target_obj = cr
                    target_is_crate = True

        if target_obj is not None:
            tpos = target_obj["pos"]
            tvel = target_obj.get("vel", [0.0, 0.0, 0.0])
            z_off = 1.5 if target_is_crate else (2.0 if target_obj.get("kind") == "npc" else 4.5)
            dist_2d = math.hypot(tpos[0] - x, tpos[1] - y)
            los_now = self.arena.has_los(muzzle, [tpos[0], tpos[1], tpos[2] + z_off])

            # Compute weapon-specific aim & fire
            best_aim_set = False
            for s_idx, w in enumerate(me["weapons"]):
                if w is None:
                    continue
                kind = w["kind"]
                wst = w["stats"]
                cd = float(w.get("cooldown", 0.0))

                if kind == "lightning":
                    if not target_is_crate and dist_2d <= float(wst["range"]) - 1.0 and energy > 12.0 and los_now:
                        lyaw = math.atan2(tpos[1] - y, tpos[0] - x)
                        lpitch = math.atan2((tpos[2] + z_off) - muzzle_z, max(dist_2d, 1.0))
                        if not best_aim_set:
                            aim_yaw, aim_pitch = lyaw, lpitch
                            best_aim_set = True
                        if cd <= 0.05 and abs(wrap_angle(me["aim"][0] - lyaw)) < math.radians(float(wst["cone_deg"]) * 0.45):
                            fire[s_idx] = True

                elif kind == "laser":
                    lrange = float(wst["range"])
                    if not target_is_crate and dist_2d <= lrange:
                        lyaw = math.atan2(tpos[1] - y, tpos[0] - x)
                        lpitch = math.atan2((tpos[2] + z_off) - muzzle_z, max(dist_2d, 1.0))
                        charge = float(w.get("charge", 0.0))
                        max_charge = float(wst["charge"])
                        min_charge = float(wst["min_charge"])
                        if charge > 0.0 or cd <= 0.15 or not best_aim_set:
                            aim_yaw, aim_pitch = lyaw, lpitch
                            best_aim_set = True
                        aim_err = abs(wrap_angle(me["aim"][0] - lyaw))
                        if charge > 0.0:
                            # Already charging: release when fully charged & lined up & LOS, or enemy about to break/short
                            ready_fire = (
                                los_now
                                and aim_err < 0.08
                                and (
                                    charge >= max_charge - 0.05
                                    or (charge >= min_charge and float(target_obj.get("hp", 9999)) <= float(wst["damage"]) * 0.5)
                                    or (charge >= min_charge and dist_2d < 46.0 and "lightning" in target_obj.get("weapons", []))
                                )
                            )
                            if ready_fire:
                                fire[s_idx] = False  # release to fire!
                            else:
                                fire[s_idx] = True   # keep holding
                        elif cd <= 0.05 and los_now:
                            fire[s_idx] = True       # start charging

                elif kind == "gun":
                    if dist_2d <= float(wst["range"]) and w.get("reload", 0.0) <= 0.0 and w.get("ammo", 0) > 0:
                        gyaw, gpitch, _ = solve_gun_lead(muzzle, tpos, tvel, float(wst["speed"]), z_off)
                        if not best_aim_set:
                            aim_yaw, aim_pitch = gyaw, gpitch
                            best_aim_set = True
                        if los_now and cd <= 0.05 and abs(wrap_angle(me["aim"][0] - gyaw)) < 0.14:
                            fire[s_idx] = True

                elif kind == "mortar":
                    if not target_is_crate and dist_2d >= 12.0 and dist_2d <= 225.0 and cd <= 0.05:
                        # Lead slightly for shell travel time
                        est_t = dist_2d / (float(wst["speed"]) * 0.8)
                        px = tpos[0] + tvel[0] * min(est_t, 2.0)
                        py = tpos[1] + tvel[1] * min(est_t, 2.0)
                        pz = tpos[2] + 1.0
                        dx = math.hypot(px - x, py - y)
                        dz = pz - (z + 9.5)
                        pitch = solve_mortar_pitch(dx, dz, float(wst["speed"]), float(wst["gravity"]), high_arc=(not los_now))
                        if pitch is not None:
                            myaw = math.atan2(py - y, px - x)
                            if not best_aim_set:
                                aim_yaw, aim_pitch = myaw, pitch
                                best_aim_set = True
                            if abs(wrap_angle(me["aim"][0] - myaw)) < 0.14:
                                fire[s_idx] = True
        else:
            # Out of combat: reload gun if low & orient front armor toward last threat or scan
            for w in me["weapons"]:
                if w and w["kind"] == "gun" and w.get("reload", 0.0) <= 0.0 and w.get("ammo", 12) < w["stats"]["mag"] * 0.6:
                    act["reload"] = True
            if self.last_threat_bearing is not None and t - self.last_threat_t < 6.0:
                aim_yaw = self.last_threat_bearing
                aim_pitch = 0.0
            else:
                aim_yaw = wrap_angle(me["aim"][0] + 0.35)
                aim_pitch = 0.0

        act["aim"] = [wrap_angle(aim_yaw), clamp(aim_pitch, self.min_pitch, self.max_pitch)]
        act["fire"] = fire
        return act


mf.run(Bot)

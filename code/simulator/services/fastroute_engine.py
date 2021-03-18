import polyline
import os
import pickle
import numpy as np
from config.hex_setting import DATA_DIR, MAX_MOVE
from common import mesh, geoutils


class FastRoutingEngine(object):
    def __init__(self):
        self.tt_map = np.load(os.path.join(DATA_DIR, 'tt_map.npy'))
        self.routes = pickle.load(open(os.path.join(DATA_DIR, 'routes.pkl'), 'rb'))

        d = self.tt_map.copy()
        for x in range(d.shape[0]):
            origin_lon = mesh.X2lon(x)
            for y in range(d.shape[1]):
                origin_lat = mesh.Y2lat(y)
                for axi in range(d.shape[2]):
                    x_ = x + axi - MAX_MOVE
                    destin_lon = mesh.X2lon(x_)
                    for ayi in range(d.shape[3]):
                        y_ = y + ayi - MAX_MOVE
                        destin_lat = mesh.Y2lat(y_)
                        d[x, y, axi, ayi] = geoutils.great_circle_distance(
                            origin_lon, origin_lat, destin_lon, destin_lat)
        self.ref_d = d  # Distance in meters

    # (Origin - destination) pairs
    def route(self, od_pairs):
        results = []
        for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
            x, y = mesh.convert_lonlat_to_xy(origin_lon, origin_lat)
            x_, y_ = mesh.convert_lonlat_to_xy(dest_lon, dest_lat)
            ax, ay = x_ - x, y_ - y
            axi = x_ - x + MAX_MOVE
            ayi = y_ - y + MAX_MOVE
            trajectory = polyline.decode(self.routes[(x, y)][(ax, ay)]) # Route from origin to destination
            triptime = self.tt_map[x, y, axi, ayi]
            results.append((trajectory, triptime))
        return results

    def route_hex(self,origin_coord, dest_coord):
        origin_lon, origin_lat = origin_coord
        dest_lon, dest_lat = dest_coord
        # for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
        x, y = mesh.convert_lonlat_to_xy(origin_lon, origin_lat)
        x_, y_ = mesh.convert_lonlat_to_xy(dest_lon, dest_lat)
        ax, ay = x_ - x, y_ - y
        axi = x_ - x + MAX_MOVE
        ayi = y_ - y + MAX_MOVE
        trajectory = polyline.decode(self.routes[(x, y)][(ax, ay)]) # Route from origin to destination
        triptime = self.tt_map[x, y, axi, ayi]#//SEC_PER_MIN
        return trajectory, triptime

    # Estimating arrival (Duration) continously until we reach destination
    def eta_many_to_many(self, origins, destins):
        origins_lon, origins_lat = zip(*origins)
        destins_lon, destins_lat = zip(*destins)
        origins_lon, origins_lat, destins_lon, destins_lat = map(np.array, [origins_lon, origins_lat, destins_lon, destins_lat])
        d = geoutils.great_circle_distance(origins_lon[:, None],origins_lat[:, None],
                                           destins_lon, destins_lat)
        return d
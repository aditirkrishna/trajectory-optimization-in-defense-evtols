from planning_layer import (
    setup_planning_layer,
    RoutePlanner,
    GraphRoutePlanner,
    EnergyOptimizer,
    RiskManager,
    MissionPlanner,
)


def main() -> None:
    config, logger = setup_planning_layer()
    planner = RoutePlanner(config)
    route = planner.optimize_route(
        start_lat=45.0,
        start_lon=-122.0,
        goal_lat=45.2,
        goal_lon=-122.3,
        start_alt_m=120.0,
        time_iso="2024-01-01T12:00:00",
    )
    energy = EnergyOptimizer(config).estimate_route_energy(route)
    risk = RiskManager(config).evaluate_route_risk(route)
    mission = MissionPlanner(config).build_single_route_mission(route)
    cost = planner.compute_route_cost(route, time_iso="2024-01-01T12:00:00")
    logger.info("Route points={} | energy_kwh={:.2f} | risk={:.2f} | cost={:.3f}", len(route), energy, risk, cost)

    # Graph-based alternatives
    graph_planner = GraphRoutePlanner(config)
    from planning_layer.routing.graph_router import GridBounds
    bounds = GridBounds(min_lat=44.9, min_lon=-122.4, max_lat=45.3, max_lon=-122.0)
    G = graph_planner.build_grid_graph(bounds, lat_steps=15, lon_steps=15, alt_m=120.0, time_iso="2024-01-01T12:00:00")
    alt_routes = graph_planner.k_shortest_routes(G, 45.0, -122.0, 45.2, -122.3, k=3)
    logger.info("Alternative routes found={}", len(alt_routes))
    for idx, r in enumerate(alt_routes, 1):
        c = planner.compute_route_cost(r, time_iso="2024-01-01T12:00:00")
        logger.info("Alt {}: points={} cost={:.3f}", idx, len(r), c)
    logger.info("Mission summary keys={}", list(mission.keys()))


if __name__ == "__main__":
    main()



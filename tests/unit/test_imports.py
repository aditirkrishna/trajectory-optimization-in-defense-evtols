def test_imports():
    from planning_layer import (
        setup_planning_layer,
        RoutePlanner,
        GraphRoutePlanner,
        EnergyOptimizer,
        RiskManager,
        MissionPlanner,
    )

    config, _ = setup_planning_layer()
    assert config.working_crs() == "EPSG:4326"

    # Build a small grid graph and compute k=1 route
    graph_planner = GraphRoutePlanner(config)
    from planning_layer.routing.graph_router import GridBounds
    bounds = GridBounds(min_lat=44.99, min_lon=-122.01, max_lat=45.01, max_lon=-121.99)
    G = graph_planner.build_grid_graph(bounds, lat_steps=5, lon_steps=5, alt_m=100.0, time_iso="2024-01-01T12:00:00")
    routes = graph_planner.k_shortest_routes(G, 45.0, -122.0, 45.01, -121.99, k=1)
    assert len(routes) >= 1



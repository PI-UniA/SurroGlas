import gmsh
def create_mesh(path: str, dim: int, name: str, t_start: int, t_end: int):
    gmsh.initialize()
    gmsh.model.add(f"Glass {dim}D mesh")

    resolution_fine = 0.1
    resolution_mid = 1.0
    resolution_coarse = 3.0

    if dim == 1:
        left = gmsh.model.occ.addPoint(0.0,0.0,0.0,resolution_fine,0)
        gmsh.model.occ.addPoint(5.0,0.0,0.0,resolution_mid,1)
        gmsh.model.occ.addPoint(25.0,0.0,0.0,resolution_coarse,2)
        gmsh.model.occ.addPoint(45.0,0.0,0.0,resolution_mid,3)
        right = gmsh.model.occ.addPoint(50.0,0.0,0.0,resolution_fine,4)

        gmsh.model.occ.addLine(0,1,0)
        gmsh.model.occ.addLine(1,2,1)
        gmsh.model.occ.addLine(2,3,2)
        gmsh.model.occ.addLine(3,4,3)

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(0, [left], 10)
        gmsh.model.addPhysicalGroup(0, [right], 12)

        gmsh.model.addPhysicalGroup(1, [0,1,2,3], 0)
        gmsh.model.setPhysicalName(1, 0, "cells")
    
    elif dim == 2:
        w = 20
        h = 4
        # width and thickness of the mesh
        # Add points for the rectangle corners
        p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, resolution_mid)
        p2 = gmsh.model.occ.addPoint(w, 0.0, 0.0, resolution_mid)
        p3 = gmsh.model.occ.addPoint(w, h, 0.0, resolution_mid)
        p4 = gmsh.model.occ.addPoint(0.0, h, 0.0, resolution_mid)

        # Create lines to form the boundaries of the rectangle
        bottom_line = gmsh.model.occ.addLine(p1, p2)
        right_line = gmsh.model.occ.addLine(p2, p3)
        top_line = gmsh.model.occ.addLine(p3, p4)
        left_line = gmsh.model.occ.addLine(p4, p1)

        # Create a loop and a plane surface
        curve_loop = gmsh.model.occ.addCurveLoop([bottom_line, right_line, top_line, left_line])
        plane_surface = gmsh.model.occ.addPlaneSurface([curve_loop])

        # Synchronize the CAD kernel with the Gmsh model
        gmsh.model.occ.synchronize()

        # Set the number of divisions (elements) along each line
        gmsh.model.mesh.setTransfiniteCurve(bottom_line, 51)  # Bottom line: nodes
        gmsh.model.mesh.setTransfiniteCurve(right_line, 11)   # Right line: nodes
        gmsh.model.mesh.setTransfiniteCurve(top_line, 51)  # Bottom line: nodes
        gmsh.model.mesh.setTransfiniteCurve(left_line, 11)   # Right line: nodes
  

        # Set a transfinite surface (ensures regular meshing)
        gmsh.model.mesh.setTransfiniteSurface(plane_surface)
        gmsh.model.mesh.setRecombine(2, plane_surface)  # Optional: make the mesh quadrilateral
        
        # Add physical groups for the boundaries
        gmsh.model.addPhysicalGroup(1, [left_line], 10)  # Physical group for the bottom line
        gmsh.model.addPhysicalGroup(1, [top_line], 11)   # Physical group for the right line
        gmsh.model.addPhysicalGroup(1, [right_line], 12)     # Physical group for the top line
        gmsh.model.addPhysicalGroup(1, [bottom_line], 13)    # Physical group for the left line
        
        gmsh.model.addPhysicalGroup(2, [plane_surface], 0)


    elif dim == 3:
        # Define zones with different dimensions and time zones
        zones = [
            {"dims": (20, 5, 0.005),"name": "A", "t_start": 0, "t_end": 90},      # Zone A
            {"dims": (30, 5, 0.005),"name": "B1","t_start": 90,"t_end": 100},      # Zone B1
            {"dims": (10, 5, 0.005),"name": "B2", "t_start": 100,"t_end": 120},     # Zone B2
            {"dims": (20, 5, 0.005),"name": "C","t_start": 120, "t_end": 130},    # Zone C
        ]

        # apply simulation over all zones
        if name == "all":
                # Filter zones based on the specified time range
                filtered_zones = [zone for zone in zones if zone["t_start"] < t_end and zone["t_end"] > t_start]
                
                if not filtered_zones:
                    raise ValueError(f"No zones found within the time range ({t_start} - {t_end}).")

                # Accumulate dimensions for the filtered zones
                accumulated_length = sum(zone["dims"][0] for zone in filtered_zones)
                width = filtered_zones[0]["dims"][1]  # Assuming width and height are the same across all zones
                height = filtered_zones[0]["dims"][2]

                accumulated_t_start = min(zone["t_start"] for zone in filtered_zones)
                accumulated_t_end = max(zone["t_end"] for zone in filtered_zones)
                
                # Adjust the specified time range if needed
                if t_start < accumulated_t_start:
                    t_start = accumulated_t_start
                if t_end > accumulated_t_end:
                    t_end = accumulated_t_end

                zone = {
                    "dims": (accumulated_length, width, height),
                    "name": "all",
                    "t_start": accumulated_t_start,
                    "t_end": accumulated_t_end
                }
        
        # apply simulation for each zone individually   
        else:
            # Find the zone that matches the provided zone_name
            zone = next((z for z in zones if z["name"] == name), None)
            if zone is None:
                raise ValueError(f"Zone {name} not found in predefined zones.")

            # Check if the specified time range falls within the zone's time interval
            if t_start < zone["t_start"] or t_end > zone["t_end"]:
                raise ValueError(f"Specified time range ({t_start} - {t_end}) is outside of Zone {name}'s time interval ({zone['t_start']} - {zone['t_end']}).")
        
        length, width, height = zone["dims"]
        name = zone["name"]

        # Create a box mesh with given dimensions and time zone
        for i, zone in enumerate(zones):

            # Define points for the box (0D)
            p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, resolution_mid)
            p2 = gmsh.model.occ.addPoint(length, 0.0, 0.0, resolution_mid)
            p3 = gmsh.model.occ.addPoint(length, width, 0.0, resolution_mid)
            p4 = gmsh.model.occ.addPoint(0.0, width, 0.0, resolution_mid)
            p5 = gmsh.model.occ.addPoint(0.0, 0.0, height, resolution_mid)
            p6 = gmsh.model.occ.addPoint(length, 0.0, height, resolution_mid)
            p7 = gmsh.model.occ.addPoint(length, width, height, resolution_mid)
            p8 = gmsh.model.occ.addPoint(0.0, width, height, resolution_mid)

            # Create lines (edges of the box) (1D)
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)
            l5 = gmsh.model.occ.addLine(p1, p5)
            l6 = gmsh.model.occ.addLine(p2, p6)
            l7 = gmsh.model.occ.addLine(p3, p7)
            l8 = gmsh.model.occ.addLine(p4, p8)
            l9 = gmsh.model.occ.addLine(p5, p6)
            l10 = gmsh.model.occ.addLine(p6, p7)
            l11 = gmsh.model.occ.addLine(p7, p8)
            l12 = gmsh.model.occ.addLine(p8, p5)

            # Create surfaces (faces of the box) (2D)
            front_face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([l1, l6, l9, l5])])
            back_face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([l3, l7, l11, l8])])
            left_face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([l4, l5, l12, l8])])
            right_face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([l2, l6, l10, l7])])
            top_face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([l9, l10, l11, l12])])
            bottom_face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])])

            # Create a volume (the box itself) (3D)
            box = gmsh.model.occ.addSurfaceLoop([front_face, back_face, left_face, right_face, top_face, bottom_face])
            volume = gmsh.model.occ.addVolume([box])

            gmsh.model.occ.synchronize()

            # Add a physical group for the current zone
            volume_tag = gmsh.model.addPhysicalGroup(3, [volume], 1000 + i)
            gmsh.model.setPhysicalName(3, 1000 + i, f"Zone_{name}")

            # Add physical groups for the faces of the box
            gmsh.model.addPhysicalGroup(2, [front_face], 100 + i)
            gmsh.model.setPhysicalName(2, 100 + i, f"Front_Face_{name}")

            gmsh.model.addPhysicalGroup(2, [back_face], 200 + i)
            gmsh.model.setPhysicalName(2, 200 + i, f"Back_Face_{name}")

            gmsh.model.addPhysicalGroup(2, [left_face], 300 + i)
            gmsh.model.setPhysicalName(2, 300 + i, f"Left_Face_{name}")

            gmsh.model.addPhysicalGroup(2, [right_face], 400 + i)
            gmsh.model.setPhysicalName(2, 400 + i, f"Right_Face_{name}")

            gmsh.model.addPhysicalGroup(2, [top_face], 500 + i)
            gmsh.model.setPhysicalName(2, 500 + i, f"Top_Face_{name}")

            gmsh.model.addPhysicalGroup(2, [bottom_face], 600 + i)
            gmsh.model.setPhysicalName(2, 600 + i, f"Bottom_Face_{name}")

            # Shift coordinates for the next zone
            #shift_x = length
            #gmsh.model.occ.translate([(3, volume)], shift_x * i, 0, 0)

    # Generate mesh
                    # Visualize the mesh
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=dim)
    gmsh.write(path)
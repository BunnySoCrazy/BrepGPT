from OCC.Core.TopoDS import (TopoDS_Shell, TopoDS_Compound,
                              topods_Face, topods_Wire, topods_Solid)
from OCC.Core.TopAbs import (TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE,
                              TopAbs_VERTEX, TopAbs_SOLID)
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Face, ShapeFix_Shell, ShapeFix_Solid


def read_step(filepath):
    reader = STEPControl_Reader()
    reader.ReadFile(filepath)
    reader.TransferRoots()
    return reader.OneShape()


def write_step(shape, filepath):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(filepath)


def extract_faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        faces.append(topods_Face(explorer.Current()))
        explorer.Next()
    return faces


def count_vertices(shape):
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def get_faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        face = topods_Face(explorer.Current())
        faces.append(face)
        explorer.Next()
    return faces


def split_face_by_inner_wires(face):
    surface = BRep_Tool.Surface(face)
    
    outer_wire_direct = breptools.OuterWire(face)
    if outer_wire_direct.IsNull():
        raise ValueError("Failed to get outer wire of face")
    
    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
    wires = []
    while wire_exp.More():
        wire = topods_Wire(wire_exp.Current())
        
        edge_count = 0
        edge_exp = TopExp_Explorer(wire, TopAbs_EDGE)
        while edge_exp.More():
            edge_count += 1
            edge_exp.Next()
        
        if edge_count < 3:
            print(f"Wire contains only {edge_count} edges, which is less than the minimum required (3)")
            return None
        
        wires.append(wire)
        wire_exp.Next()
        
    if len(wires) <= 1:
        return [{"face": face, "type": "outer"}]
    
    outer_wire = None
    inner_wires = []
    
    for wire in wires:
        if wire.IsSame(outer_wire_direct):
            outer_wire = wire
        else:
            inner_wires.append(wire)
    
    if outer_wire is None:
        raise ValueError("Cannot find matching outer wire in collected wires")
    
    new_faces = []
    outer_face = BRepBuilderAPI_MakeFace(surface, outer_wire).Face()
    new_faces.append({"face": outer_face, "type": "outer"})
    
    for inner_wire in inner_wires:
        inner_face = BRepBuilderAPI_MakeFace(surface, inner_wire).Face().Reversed()
        new_faces.append({"face": inner_face, "type": "inner"})
    
    return new_faces


def sew_faces_to_solid(faces, tolerance=1e-6):
    sewer = BRepBuilderAPI_Sewing(tolerance)
    
    for face in faces:
        sewer.Add(face['face'])
    
    sewer.Perform()
    sewn_shape = sewer.SewedShape()
    
    shape_fixer = ShapeFix_Shape(sewn_shape)
    shape_fixer.SetPrecision(tolerance)
    shape_fixer.SetMaxTolerance(tolerance)
    shape_fixer.Perform()
    fixed_shape = shape_fixer.Shape()
    
    shell = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(shell)
    
    exp = TopExp_Explorer(fixed_shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        face_fixer = ShapeFix_Face(face)
        face_fixer.Perform()
        fixed_face = face_fixer.Face()
        builder.Add(shell, fixed_face)
        exp.Next()
    
    shell_fixer = ShapeFix_Shell(shell)
    shell_fixer.FixFaceOrientation(shell)
    shell_fixer.Perform()
    fixed_shell = shell_fixer.Shell()
    
    solid_maker = BRepBuilderAPI_MakeSolid()
    solid_maker.Add(fixed_shell)
    solid = solid_maker.Solid()
    
    solid_fixer = ShapeFix_Solid(solid)
    solid_fixer.Perform()
    fixed_solid = solid_fixer.Solid()
    
    final_fixer = ShapeFix_Shape(fixed_solid)
    final_fixer.SetPrecision(tolerance)
    final_fixer.SetMaxTolerance(tolerance)
    final_fixer.Perform()
    
    final_shape = final_fixer.Shape()

    solid_explorer = TopExp_Explorer(final_shape, TopAbs_SOLID)
    solid_count = 0
    solids = []
    
    while solid_explorer.More():
        solid = topods_Solid(solid_explorer.Current())
        solids.append(solid)
        solid_count += 1
        solid_explorer.Next()
    
    fixed_solids = []
    for i, solid in enumerate(solids):
        solid_fixer = ShapeFix_Solid(solid)
        solid_fixer.Perform()
        fixed_solid = solid_fixer.Solid()
        
        final_solid_fixer = ShapeFix_Shape(fixed_solid)
        final_solid_fixer.SetPrecision(tolerance)
        final_solid_fixer.SetMaxTolerance(tolerance)
        final_solid_fixer.Perform()
        
        fixed_solids.append(final_solid_fixer.Shape())
    
    if len(fixed_solids) == 1:
        return fixed_solids[0]
    
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    
    for solid in fixed_solids:
        builder.Add(compound, solid)
    
    return compound


def sew_faces_to_solid_no_fix(faces, tolerance=1e-6):
    sewer = BRepBuilderAPI_Sewing(tolerance)
    
    for face in faces:
        sewer.Add(face['face'])
    
    sewer.Perform()
    sewn_shape = sewer.SewedShape()
    
    shell = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(shell)
    
    exp = TopExp_Explorer(sewn_shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        builder.Add(shell, face)
        exp.Next()
    
    solid_maker = BRepBuilderAPI_MakeSolid()
    solid_maker.Add(shell)
    solid = solid_maker.Solid()
    
    return solid

def save_shape_to_stl(shape, output_file="output_mesh.stl", mesh_precision=0.1):
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    
    mesh = BRepMesh_IncrementalMesh(shape, mesh_precision)
    mesh.Perform()
    
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, output_file)
    print(f"Saved STL mesh file: {output_file}")

use pyo3::prelude::*;

type Colorf = Vec<f32>;
type Vec2 = (f32, f32);
type Triangle = (Vec2, Vec2, Vec2);

fn line(a: Vec2, b: Vec2, x: Vec2) -> f32 {
    let (x0, y0) = a;
    let (x1, y1) = b;
    let (x, y) = x;
    (y1 - y0) * x -
        (x1 - x0) * y +
        (x1 - x0) * y0 -
        (y1 - y0) * x0
}

fn inside_triangle(triangle: Triangle, p: Vec2) -> bool {
    let p = (p.0 + 0.5, p.1 + 0.5);

    let (a, b, c) = triangle;
    line(a, b, p) >= 0.0 &&
        line(b, c, p) >= 0.0 &&
        line(c, a, p) >= 0.0
}

#[pyfunction]
fn render_triangle(
    vertices: Vec<f32>,
    width: usize,
    height: usize,
    color: Colorf,
    gpu: &Bound<'_, PyAny>,
) -> PyResult<()> {   
    let color = (
        (color[0] * 255.0) as u8,
        (color[1] * 255.0) as u8,
        (color[2] * 255.0) as u8,
    );
    let draw_pixel = gpu.getattr("draw_pixel")?;                       
    let rgb8 = gpu.getattr("RGB8")?;

    for verts in vertices.chunks(6) {
        let a = (verts[0], verts[1]);
        let b = (verts[2], verts[3]);
        let c = (verts[4], verts[5]);

        let points = [a, b, c];

        let min_x = points.iter().map(|f| f.0).min_by(|a, b| a.total_cmp(b)).unwrap();
        let max_x = points.iter().map(|f| f.0).max_by(|a, b| a.total_cmp(b)).unwrap();

        let min_y = points.iter().map(|f| f.1).min_by(|a, b| a.total_cmp(b)).unwrap();
        let max_y = points.iter().map(|f| f.1).max_by(|a, b| a.total_cmp(b)).unwrap();

        let min_x = min_x.max(0.0) as usize;
        let min_y = min_y.max(0.0) as usize;

        let max_x = max_x.min(width as f32 - 1.0) as usize;
        let max_y = max_y.min(height as f32 - 1.0) as usize;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if !inside_triangle((a, b, c), (x as f32, y as f32)) {
                    continue;
                }

                draw_pixel.call1((
                    (x, y),
                    &rgb8,
                    color,
                ))?;
            }
        }
    }
    
    Ok(())
}

#[pymodule]
fn rendererrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_triangle, m)?)?;

    Ok(())
}

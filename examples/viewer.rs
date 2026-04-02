use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4Swizzles};
use loke::{self, Adaptor, ScalarAdaptor, TrigonometryAdaptor};
use std::{marker::PhantomData, ops::Range};

// ---------- Glam adaptor ----------

#[derive(Clone)]
struct GlamAdaptor;

impl ScalarAdaptor for GlamAdaptor {
    type Float = f32;
    #[inline(always)]
    fn abs(x: f32) -> f32 {
        x.abs()
    }
    #[inline(always)]
    fn sqrt(x: f32) -> f32 {
        x.sqrt()
    }
    #[inline(always)]
    fn is_finite(x: f32) -> bool {
        x.is_finite()
    }
    #[inline(always)]
    fn mul_add(x: f32, a: f32, b: f32) -> f32 {
        x.mul_add(a, b)
    }
    #[inline(always)]
    fn scalar(val: f64) -> f32 {
        val as f32
    }
    #[inline(always)]
    fn epsilon() -> f32 {
        f32::EPSILON
    }
    #[inline(always)]
    fn min(x: f32, other: f32) -> f32 {
        x.min(other)
    }
    #[inline(always)]
    fn max(x: f32, other: f32) -> f32 {
        x.max(other)
    }
    #[inline(always)]
    fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
        x.clamp(lo, hi)
    }

    #[inline(always)]
    fn ceil(x: Self::Float) -> Self::Float {
        x.ceil()
    }

    #[inline(always)]
    fn to_usize(x: Self::Float) -> usize {
        x as usize
    }
}

impl TrigonometryAdaptor for GlamAdaptor {
    #[inline(always)]
    fn acos(x: Self::Float) -> Self::Float {
        x.acos()
    }

    #[inline(always)]
    fn sin_cos(x: Self::Float) -> (Self::Float, Self::Float) {
        x.sin_cos()
    }

    #[inline(always)]
    fn sin(x: Self::Float) -> Self::Float {
        x.sin()
    }

    #[inline(always)]
    fn cos(x: Self::Float) -> Self::Float {
        x.cos()
    }

    #[inline(always)]
    fn tan(x: Self::Float) -> Self::Float {
        x.tan()
    }
}

impl Adaptor<3> for GlamAdaptor {
    type Vector = Vec3;
    type Scalar = f32;

    #[inline(always)]
    fn zero_vector() -> Vec3 {
        Vec3::ZERO
    }
    #[inline(always)]
    fn vector(coords: [f32; 3]) -> Vec3 {
        Vec3::new(coords[0], coords[1], coords[2])
    }
    #[inline(always)]
    fn vector_coord(v: Vec3, i: usize) -> f32 {
        v[i]
    }

    #[inline(always)]
    fn vector_length(v: Vec3) -> f32 {
        v.length()
    }

    #[inline(always)]
    fn vector_length_sq(v: Self::Vector) -> Self::Scalar {
        v.length_squared()
    }

    #[inline(always)]
    fn normalize(v: Self::Vector) -> Self::Vector {
        v.normalize()
    }

    #[inline(always)]
    fn dot_product(a: Self::Vector, b: Self::Vector) -> Self::Scalar {
        a.dot(b)
    }

    #[inline(always)]
    fn coord_arr(v: Self::Vector) -> [Self::Scalar; 3] {
        [v.x, v.y, v.z]
    }
}

type Spline = loke::Spline<3, GlamAdaptor>;
type Arc = loke::Arc<3, GlamAdaptor>;

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// ---------- GPU types ----------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 4],
    arc_len: f32,
}

impl Vertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32],
    };
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    dash_length: f32,
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PointInstance {
    center: [f32; 3],
    color: [f32; 4],
}

impl PointInstance {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
    };
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GizmoAxisInstance {
    start: [f32; 3],
    end: [f32; 3],
    color: [f32; 4],
}

impl GizmoAxisInstance {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
    };
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const BG: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
const MSAA_SAMPLES: u32 = 4;
const POINT_SIZE: f32 = 8.0;
const FOV: f32 = std::f32::consts::FRAC_PI_4;
const GIZMO_INNER: f32 = 1.1;
const GIZMO_OUTER: f32 = 4.5;
const GIZMO_HIT_THRESHOLD: f32 = 12.0;

const AXIS_DIRS: [Vec3; 3] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
];
const AXIS_COLORS: [[f32; 4]; 3] = [
    [1.0, 0.2, 0.2, 1.0],
    [0.2, 1.0, 0.2, 1.0],
    [0.3, 0.3, 1.0, 1.0],
];

const SPLINE_COLOR: [f32; 4] = [0.9, 0.9, 0.85, 1.0];
const CTRL_CAGE_COLOR: [f32; 4] = [0.7, 0.35, 0.3, 0.8];
const BBOX_COLOR: [f32; 4] = [0.35, 0.3, 0.9, 0.3];
const POINT_COLOR: [f32; 4] = [1.0, 0.4, 0.1, 1.0];
const INPUT_POINT_COLOR: [f32; 4] = [0.8, 0.8, 0.75, 1.0];

// ---------- Scene trait ----------

trait Scene {
    fn init() -> Box<[Vec3]>;
    fn update(
        inputs: &[Vec3],
        splines: &mut Vec<Spline>,
        arcs: &mut Vec<Arc>,
        points: &mut Vec<Vec3>,
    );
}

// ---------- Draggable point ----------

struct DraggablePoint {
    id: usize,
    position: Vec3,
}

struct DragState {
    point_idx: usize,
    axis: usize,
}

// ---------- Camera ----------

struct Camera {
    target: Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
}

impl Camera {
    fn new(center: Vec3, radius: f32) -> Self {
        Self {
            target: center,
            distance: if radius > 0.0 { radius * 2.5 } else { 5.0 },
            azimuth: std::f32::consts::FRAC_PI_4,
            elevation: 0.4,
        }
    }

    fn eye(&self) -> Vec3 {
        let ce = self.elevation.cos();
        self.target
            + self.distance
                * Vec3::new(
                    ce * self.azimuth.cos(),
                    ce * self.azimuth.sin(),
                    self.elevation.sin(),
                )
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Z);
        let proj = Mat4::perspective_rh(FOV, aspect, self.distance * 0.01, self.distance * 100.0);
        proj * view
    }

    fn right(&self) -> Vec3 {
        let forward = (self.target - self.eye()).normalize();
        forward.cross(Vec3::Z).normalize()
    }

    fn up(&self) -> Vec3 {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        right.cross(forward).normalize()
    }

    fn zoom_extents(&mut self, bbox: (Vec3, Vec3)) {
        self.target = (bbox.0 + bbox.1) * 0.5;
        let radius = (bbox.1 - bbox.0).length() * 0.5;
        self.distance = if radius > 0.0 { radius * 2.5 } else { 5.0 };
    }

    fn pixel_world(&self, screen_h: f32) -> f32 {
        self.distance * 2.0 * (FOV * 0.5).tan() / screen_h
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * 0.005;
        self.elevation += dy * 0.005;
        let limit = 89.0_f32.to_radians();
        self.elevation = self.elevation.clamp(-limit, limit);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let scale = self.distance * 0.002;
        self.target -= self.right() * dx * scale;
        self.target += self.up() * dy * scale;
    }

    fn zoom(&mut self, scroll: f32) {
        let factor = 1.0 - scroll * 0.1;
        self.distance = (self.distance * factor).max(0.001);
    }
}

// ---------- GPU resource bundle ----------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    point_buf: wgpu::Buffer,
    point_count: u32,
    gizmo_buf: wgpu::Buffer,
    gizmo_capacity: u32,
}

// ---------- App ----------

struct App<S: Scene> {
    gpu: Option<Gpu>,
    window: Option<Window>,
    // Scene I/O
    inputs: Box<[Vec3]>,
    out_splines: Vec<Spline>,
    out_arcs: Vec<Arc>,
    out_points: Vec<Vec3>,
    // Render data built from scene outputs
    vertices: Vec<Vertex>,
    points: Vec<PointInstance>,
    line_strips: Vec<std::ops::Range<u32>>,
    // Interaction
    draggable: Vec<DraggablePoint>,
    drag: Option<DragState>,
    buffers_dirty: bool,
    camera: Camera,
    // Input state
    rmb: bool,
    mmb: bool,
    lmb: bool,
    shift: bool,
    last_mouse: Option<(f32, f32)>,
    mouse_pos: (f32, f32),
    _phantom: PhantomData<S>,
}

impl<S: Scene> App<S> {
    fn new() -> Self {
        let inputs = S::init();
        let mut out_splines = Vec::new();
        let mut out_arcs = Vec::new();
        let mut out_points = Vec::new();
        S::update(&inputs, &mut out_splines, &mut out_arcs, &mut out_points);

        let draggable = inputs
            .iter()
            .enumerate()
            .map(|(i, p)| DraggablePoint {
                id: i,
                position: *p,
            })
            .collect();

        let mut app = Self {
            gpu: None,
            window: None,
            inputs,
            out_splines,
            out_arcs,
            out_points,
            vertices: Vec::new(),
            points: Vec::new(),
            line_strips: Vec::new(),
            draggable,
            drag: None,
            buffers_dirty: false,
            camera: Camera::new(Vec3::ZERO, 1.0),
            rmb: false,
            mmb: false,
            lmb: false,
            shift: false,
            last_mouse: None,
            mouse_pos: (0.0, 0.0),
            _phantom: PhantomData,
        };
        app.rebuild_geometry();
        let bbox = bounding_box(&app.vertices, &app.points);
        let center = (bbox.0 + bbox.1) * 0.5;
        let radius = (bbox.1 - bbox.0).length() * 0.5;
        app.camera = Camera::new(center, radius);
        app
    }

    fn run() {
        env_logger::builder()
            .filter_level(log::LevelFilter::Warn)
            .init();
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = Self::new();
        event_loop.run_app(&mut app).unwrap();
    }

    fn rebuild_geometry(&mut self) {
        self.vertices.clear();
        self.points.clear();
        self.line_strips.clear();

        for spline in &self.out_splines {
            let cps = spline.control_points();
            let start = self.vertices.len() as u32;
            // Solid curve (arc_len=0 → never dashed)
            self.vertices
                .extend(spline.adaptive_samples(0.001).map(|p| Vertex {
                    pos: p.into(),
                    color: SPLINE_COLOR,
                    arc_len: 0.0,
                }));
            // Control cage back to front (dashed)
            let n = cps.len();
            let mut arc = 0.0_f32;
            for i in (0..n).rev() {
                if i < n - 1 {
                    let d = cps[i] - cps[i + 1];
                    arc += d.length();
                }
                self.vertices.push(Vertex {
                    pos: cps[i].into(),
                    color: CTRL_CAGE_COLOR,
                    arc_len: arc,
                });
            }
            let end = self.vertices.len() as u32;
            self.line_strips.push(start..end);
            let (min, max) = spline.bounds();
            rebuild_bbox(min, max, &mut self.vertices, &mut self.line_strips);
        }

        for arc in &self.out_arcs {
            let start = self.vertices.len() as u32;
            self.vertices
                .extend(arc.adaptive_samples(0.001).map(|p| Vertex {
                    pos: p.into(),
                    color: SPLINE_COLOR,
                    arc_len: 0.0,
                }));
            let end = self.vertices.len() as u32;
            self.line_strips.push(start..end);
            let (min, max) = arc.bounds();
            rebuild_bbox(min, max, &mut self.vertices, &mut self.line_strips);
        }

        // Input points (gizmo centers)
        for p in self.inputs.iter() {
            self.points.push(PointInstance {
                center: (*p).into(),
                color: INPUT_POINT_COLOR,
            });
        }

        // Scene output points
        for p in &self.out_points {
            self.points.push(PointInstance {
                center: (*p).into(),
                color: POINT_COLOR,
            });
        }
    }

    fn zoom_extents(&mut self) {
        let bbox = bounding_box(&self.vertices, &self.points);
        self.camera.zoom_extents(bbox);
        self.request_redraw();
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn world_to_screen(&self, vp: &Mat4, w: f32, h: f32, world: Vec3) -> Option<(f32, f32)> {
        let clip = *vp * world.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.xy() / clip.w;
        Some(((ndc.x * 0.5 + 0.5) * w, (0.5 - ndc.y * 0.5) * h))
    }

    fn build_gizmo_instances(&self, screen_h: f32) -> Vec<GizmoAxisInstance> {
        let pw = self.camera.pixel_world(screen_h);
        let inner = POINT_SIZE * pw * GIZMO_INNER;
        let outer = POINT_SIZE * pw * GIZMO_OUTER;
        let mut axes = Vec::with_capacity(self.draggable.len() * 3);
        for dp in &self.draggable {
            for ax in 0..3 {
                let dir = AXIS_DIRS[ax];
                axes.push(GizmoAxisInstance {
                    start: (dp.position + dir * inner).into(),
                    end: (dp.position + dir * outer).into(),
                    color: AXIS_COLORS[ax],
                });
            }
        }
        axes
    }

    fn hit_test_gizmo(&self, mx: f32, my: f32, screen_h: f32) -> Option<(usize, usize)> {
        let gpu = self.gpu.as_ref()?;
        let w = gpu.config.width as f32;
        let h = gpu.config.height as f32;
        let vp = self.camera.view_proj(w / h);
        let pw = self.camera.pixel_world(screen_h);
        let inner = POINT_SIZE * pw * GIZMO_INNER;
        let outer = POINT_SIZE * pw * GIZMO_OUTER;
        let mut best_dist = GIZMO_HIT_THRESHOLD;
        let mut best = None;
        for (pi, dp) in self.draggable.iter().enumerate() {
            for ax in 0..3 {
                let dir = AXIS_DIRS[ax];
                let a = self.world_to_screen(&vp, w, h, dp.position + dir * inner);
                let b = self.world_to_screen(&vp, w, h, dp.position + dir * outer);
                if let (Some(a), Some(b)) = (a, b) {
                    let d = dist_point_to_segment(mx, my, a.0, a.1, b.0, b.1);
                    if d < best_dist {
                        best_dist = d;
                        best = Some((pi, ax));
                    }
                }
            }
        }
        best
    }

    fn handle_gizmo_drag(&mut self, mx: f32, my: f32, last_mx: f32, last_my: f32) {
        let Some(DragState {
            point_idx: pi,
            axis,
        }) = self.drag
        else {
            return;
        };
        let Some(gpu) = &self.gpu else { return };
        let w = gpu.config.width as f32;
        let h = gpu.config.height as f32;
        let vp = self.camera.view_proj(w / h);
        let pos = self.draggable[pi].position;
        let axis_dir = AXIS_DIRS[axis];
        let center_s = self.world_to_screen(&vp, w, h, pos);
        let tip_s = self.world_to_screen(&vp, w, h, pos + axis_dir);
        let (Some(cs), Some(ts)) = (center_s, tip_s) else {
            return;
        };
        let ax_sx = ts.0 - cs.0;
        let ax_sy = ts.1 - cs.1;
        let ax_len = (ax_sx * ax_sx + ax_sy * ax_sy).sqrt();
        if ax_len < 0.001 {
            return;
        }
        let dmx = mx - last_mx;
        let dmy = my - last_my;
        let delta_along = (dmx * ax_sx + dmy * ax_sy) / ax_len;
        let world_delta = delta_along / ax_len;
        let new_pos = pos + axis_dir * world_delta;

        let id = self.draggable[pi].id;
        self.draggable[pi].position = new_pos;
        self.inputs[id] = new_pos;
        S::update(
            &self.inputs,
            &mut self.out_splines,
            &mut self.out_arcs,
            &mut self.out_points,
        );
        self.rebuild_geometry();
        self.buffers_dirty = true;
    }

    fn reupload_buffers(&mut self) {
        if !self.buffers_dirty {
            return;
        }
        self.buffers_dirty = false;
        let Some(gpu) = &mut self.gpu else { return };
        if !self.vertices.is_empty() {
            let data = bytemuck::cast_slice(&self.vertices);
            if data.len() as u64 > gpu.vertex_buf.size() {
                gpu.vertex_buf = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vertices"),
                        contents: data,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
            } else {
                gpu.queue.write_buffer(&gpu.vertex_buf, 0, data);
            }
            gpu.vertex_count = self.vertices.len() as u32;
        }
        if !self.points.is_empty() {
            let data = bytemuck::cast_slice(&self.points);
            if data.len() as u64 > gpu.point_buf.size() {
                gpu.point_buf = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("points"),
                        contents: data,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
            } else {
                gpu.queue.write_buffer(&gpu.point_buf, 0, data);
            }
            gpu.point_count = self.points.len() as u32;
        }
    }
}

fn rebuild_bbox(
    min: Vec3,
    max: Vec3,
    vertices: &mut Vec<Vertex>,
    line_strips: &mut Vec<Range<u32>>,
) {
    let start = vertices.len() as u32;
    vertices.extend(
        [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
        ]
        .iter()
        .map(|p| Vertex {
            pos: (*p).into(),
            color: BBOX_COLOR,
            arc_len: 0.0,
        }),
    );
    line_strips.push(start..(vertices.len() as u32));
}

// ---------- Helpers ----------

fn bounding_box(verts: &[Vertex], points: &[PointInstance]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for v in verts {
        let p = Vec3::from(v.pos);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    for pt in points {
        let p = Vec3::from(pt.center);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    if lo.x > hi.x {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    (lo, hi)
}

fn dist_point_to_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;
    let len2 = abx * abx + aby * aby;
    if len2 < 1e-8 {
        return (apx * apx + apy * apy).sqrt();
    }
    let t = ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0);
    let dx = px - (ax + t * abx);
    let dy = py - (ay + t * aby);
    (dx * dx + dy * dy).sqrt()
}

fn create_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn create_msaa_view(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
    buffers: &[wgpu::VertexBufferLayout],
    topology: wgpu::PrimitiveTopology,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

// ---------- winit ApplicationHandler ----------

impl<S: Scene> ApplicationHandler for App<S> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("loke viewer")
                    .with_inner_size(winit::dpi::PhysicalSize::new(1200, 800)),
            )
            .unwrap();

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(&window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            ..Default::default()
        }))
        .expect("Failed to create device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let config = {
            let mut c = surface
                .get_default_config(&adapter, size.width, size.height)
                .expect("Surface not supported");
            c.present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::AutoNoVsync
            };
            c
        };
        surface.configure(&device, &config);

        let depth_view = create_depth_view(&device, size.width, size.height);
        let msaa_view = create_msaa_view(&device, size.width, size.height, config.format);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                dash_length: 0.0,
                point_size: POINT_SIZE,
                screen_width: size.width as f32,
                screen_height: size.height as f32,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Pipelines
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("line.wgsl").into()),
        });
        let line_pipeline = create_pipeline(
            &device,
            &line_shader,
            &bind_group_layout,
            config.format,
            &[Vertex::LAYOUT],
            wgpu::PrimitiveTopology::LineStrip,
            true,
        );

        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("point.wgsl").into()),
        });
        let point_pipeline = create_pipeline(
            &device,
            &point_shader,
            &bind_group_layout,
            config.format,
            &[PointInstance::LAYOUT],
            wgpu::PrimitiveTopology::TriangleList,
            true,
        );

        let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gizmo shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gizmo.wgsl").into()),
        });
        let gizmo_pipeline = create_pipeline(
            &device,
            &gizmo_shader,
            &bind_group_layout,
            config.format,
            &[GizmoAxisInstance::LAYOUT],
            wgpu::PrimitiveTopology::TriangleList,
            false,
        );

        // Buffers
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: if self.vertices.is_empty() {
                &[0u8; std::mem::size_of::<Vertex>()]
            } else {
                bytemuck::cast_slice(&self.vertices)
            },
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let vertex_count = self.vertices.len() as u32;

        let point_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"),
            contents: if self.points.is_empty() {
                &[0u8; std::mem::size_of::<PointInstance>()]
            } else {
                bytemuck::cast_slice(&self.points)
            },
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let point_count = self.points.len() as u32;

        let gizmo_capacity = (self.draggable.len() * 3).max(1) as u32;
        let gizmo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo"),
            size: (gizmo_capacity as usize * std::mem::size_of::<GizmoAxisInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let surface =
            unsafe { std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) };

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            depth_view,
            msaa_view,
            line_pipeline,
            point_pipeline,
            gizmo_pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            vertex_count,
            point_buf,
            point_count,
            gizmo_buf,
            gizmo_capacity,
        });
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        self.shift = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyZ | KeyCode::Home) if pressed => {
                        self.zoom_extents();
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed {
                            let (mx, my) = self.mouse_pos;
                            let screen_h = self
                                .gpu
                                .as_ref()
                                .map(|g| g.config.height as f32)
                                .unwrap_or(800.0);
                            if let Some((pi, ax)) = self.hit_test_gizmo(mx, my, screen_h) {
                                self.drag = Some(DragState {
                                    point_idx: pi,
                                    axis: ax,
                                });
                                self.lmb = true;
                                self.last_mouse = Some((mx, my));
                            }
                        } else {
                            self.lmb = false;
                            self.drag = None;
                            self.last_mouse = None;
                        }
                    }
                    MouseButton::Right => {
                        self.rmb = pressed;
                        if !pressed {
                            self.last_mouse = None;
                        }
                    }
                    MouseButton::Middle => {
                        self.mmb = pressed;
                        if !pressed {
                            self.last_mouse = None;
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                self.mouse_pos = (x, y);
                if self.lmb && self.drag.is_some() {
                    if let Some((lx, ly)) = self.last_mouse {
                        self.handle_gizmo_drag(x, y, lx, ly);
                        self.request_redraw();
                    }
                    self.last_mouse = Some((x, y));
                } else if self.rmb || self.mmb {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = x - lx;
                        let dy = y - ly;
                        if self.mmb || (self.rmb && self.shift) {
                            self.camera.pan(dx, dy);
                        } else {
                            self.camera.orbit(dx, dy);
                        }
                        self.request_redraw();
                    }
                    self.last_mouse = Some((x, y));
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera.zoom(scroll);
                self.request_redraw();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = &mut self.gpu {
                    if new_size.width > 0 && new_size.height > 0 {
                        gpu.config.width = new_size.width;
                        gpu.config.height = new_size.height;
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        gpu.depth_view =
                            create_depth_view(&gpu.device, new_size.width, new_size.height);
                        gpu.msaa_view = create_msaa_view(
                            &gpu.device,
                            new_size.width,
                            new_size.height,
                            gpu.config.format,
                        );
                        self.request_redraw();
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                self.reupload_buffers();
                let Some(gpu) = &self.gpu else { return };
                let w = gpu.config.width as f32;
                let h = gpu.config.height as f32;
                let vp = self.camera.view_proj(w / h);
                let uniforms = Uniforms {
                    view_proj: vp.to_cols_array_2d(),
                    dash_length: 0.3,
                    point_size: POINT_SIZE,
                    screen_width: w,
                    screen_height: h,
                };
                gpu.queue
                    .write_buffer(&gpu.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

                let gizmo_instances = self.build_gizmo_instances(h);
                let gizmo_count = gizmo_instances.len() as u32;
                if gizmo_count > 0 && gizmo_count <= gpu.gizmo_capacity {
                    gpu.queue.write_buffer(
                        &gpu.gizmo_buf,
                        0,
                        bytemuck::cast_slice(&gizmo_instances),
                    );
                }

                let frame = match gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => {
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut enc = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("render"),
                    });
                {
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &gpu.msaa_view,
                            resolve_target: Some(&view),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(BG),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &gpu.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    // Line strips (one draw per spline)
                    pass.set_pipeline(&gpu.line_pipeline);
                    pass.set_bind_group(0, Some(&gpu.bind_group), &[]);
                    pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
                    for strip in &self.line_strips {
                        pass.draw(strip.clone(), 0..1);
                    }
                    // Points
                    if gpu.point_count > 0 {
                        pass.set_pipeline(&gpu.point_pipeline);
                        pass.set_bind_group(0, Some(&gpu.bind_group), &[]);
                        pass.set_vertex_buffer(0, gpu.point_buf.slice(..));
                        pass.draw(0..6, 0..gpu.point_count);
                    }
                    // Gizmos
                    if gizmo_count > 0 {
                        pass.set_pipeline(&gpu.gizmo_pipeline);
                        pass.set_bind_group(0, Some(&gpu.bind_group), &[]);
                        pass.set_vertex_buffer(0, gpu.gizmo_buf.slice(..));
                        pass.draw(0..6, 0..gizmo_count);
                    }
                }
                gpu.queue.submit(Some(enc.finish()));
                frame.present();
                if self.rmb || self.mmb || self.lmb {
                    self.request_redraw();
                }
            }
            _ => {}
        }
    }
}

// ---------- Scenes ----------

struct CurveScene;

impl Scene for CurveScene {
    fn init() -> Box<[Vec3]> {
        Box::new([
            // Spline points.
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(-0.5, 2.0, 1.0),
            Vec3::new(0.5, -2.0, 1.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(3.5, 0.0, 0.0),
            // Arc - three points.
            Vec3::new(-3.5, 0.0, 0.0), // start
            Vec3::new(-3.0, 1.5, 0.0), // middle
            Vec3::new(-2.5, 0.0, 0.0), // end
            // Arc - start-tangent-end.
            Vec3::new(3.5, 0.0, 0.0),  // start
            Vec3::new(3.6, 0.25, 0.0), // tangent-point
            Vec3::new(5.5, 0.0, 0.0),  // end
            // Arc - center-normal-start-angle
            Vec3::new(7.0, 0.0, 0.0), // axis-start
            Vec3::new(8.0, 0.0, 0.0), // axis-end
            Vec3::new(7.0, 3.0, 0.0), // arc-start
        ])
    }

    fn update(
        inputs: &[Vec3],
        splines: &mut Vec<Spline>,
        arcs: &mut Vec<Arc>,
        points: &mut Vec<Vec3>,
    ) {
        // Splines.
        splines.clear();
        let curve = Spline::create_clamped(inputs[0..5].to_vec(), 3).unwrap();
        points.clear();
        points.push({
            let (a, b) = curve.domain();
            curve.point((a + b) * 0.5).unwrap()
        });
        splines.push(curve);
        // Arcs.
        arcs.clear();
        if let Ok(arc) = Arc::from_three_points(inputs[5], inputs[6], inputs[7]) {
            arcs.push(arc);
        }
        match Arc::from_start_tangent_end(inputs[8], inputs[9] - inputs[8], inputs[10]) {
            Ok(arc) => arcs.push(arc),
            Err(e) => eprintln!("{e:?}"),
        }
        match Arc::from_axis_start_angle(
            (inputs[11], inputs[12]),
            inputs[13],
            std::f32::consts::FRAC_PI_2,
        ) {
            Ok(arc) => arcs.push(arc),
            Err(e) => eprintln!("Arc creation failed: {e:?}"),
        }
    }
}

// ---------- Entry point ----------

fn main() {
    App::<CurveScene>::run();
}

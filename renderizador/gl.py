#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
import random

type Vec2 = tuple[float, float]
type Triangle = tuple[Vec2, Vec2, Vec2]
type Color = tuple[int, int, int]

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    logical_width = 800
    logical_height = 600
    supersample = 1
    model_matrix = np.identity(4, dtype=float)
    matrix_stack = []
    lights = []
    camera_position = np.array([0.0, 0.0, 10.0], dtype=float)
    camera_matrix = np.identity(4, dtype=float)
    _time_sensors = {}

    @staticmethod
    def setup(width, height, near=0.01, far=1000, supersample=1):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.supersample = max(1, int(supersample))
        GL.logical_width = width / GL.supersample
        GL.logical_height = height / GL.supersample
        # Cada nova cena comeca sem transformacoes de modelo acumuladas.
        GL.model_matrix = np.identity(4, dtype=float)
        GL.matrix_stack = []
        GL.lights = []
        GL.camera_position = np.array([0.0, 0.0, 10.0], dtype=float)
        GL.camera_matrix = np.identity(4, dtype=float)
        GL._time_sensors = {}

    @staticmethod
    def _base_color(colors):
        """Retorna a cor base em RGB 0..255 para um material X3D."""
        emissive = np.asarray(colors.get("emissiveColor", [0.0, 0.0, 0.0]), dtype=float)
        if np.max(np.abs(emissive)) > 1e-8:
            return np.clip(emissive, 0.0, 1.0) * 255.0
        diffuse = np.asarray(colors.get("diffuseColor", [0.8, 0.8, 0.8]), dtype=float)
        return np.clip(diffuse, 0.0, 1.0) * 255.0

    @staticmethod
    def _normalize(vector, fallback=None):
        """Normaliza um vetor e evita divisões por zero."""
        value = np.asarray(vector, dtype=float)
        length = np.linalg.norm(value)
        if length > 1e-12:
            return value / length
        if fallback is None:
            return np.zeros_like(value)
        return np.asarray(fallback, dtype=float)

    @staticmethod
    def _material_color(colors, world_position, normal, albedo=None):
        """Calcula a cor de um fragmento usando o modelo de iluminação X3D."""
        emissive = np.clip(
            np.asarray(colors.get("emissiveColor", [0.0, 0.0, 0.0]), dtype=float),
            0.0,
            1.0,
        )
        diffuse = np.clip(
            np.asarray(colors.get("diffuseColor", [0.8, 0.8, 0.8]), dtype=float),
            0.0,
            1.0,
        )
        if albedo is not None:
            diffuse = np.clip(np.asarray(albedo, dtype=float), 0.0, 1.0)

        normal = GL._normalize(normal)
        view_direction = GL._normalize(GL.camera_position - world_position)
        material_ambient = np.clip(float(colors.get("ambientIntensity", 0.2)), 0.0, 1.0)
        shininess = max(1.0, np.clip(float(colors.get("shininess", 0.2)), 0.0, 1.0) * 128.0)
        specular = np.clip(
            np.asarray(colors.get("specularColor", [0.0, 0.0, 0.0]), dtype=float),
            0.0,
            1.0,
        )

        result = emissive.copy()
        if not GL.lights:
            # Mantém materiais visíveis em cenas sem iluminação declarada.
            return np.clip(result + diffuse, 0.0, 1.0) * 255.0

        for light in GL.lights:
            light_color = light["color"] * light["intensity"]
            result += diffuse * material_ambient * light["ambientIntensity"] * light_color

            if light["type"] == "directional":
                light_direction = -light["direction"]
            else:
                light_direction = light["position"] - world_position
                distance = np.linalg.norm(light_direction)
                if distance > 1e-12:
                    light_direction = light_direction / distance
                else:
                    light_direction = normal

            diffuse_factor = max(0.0, float(np.dot(normal, light_direction)))
            result += diffuse * diffuse_factor * light_color

            if diffuse_factor > 0.0 and np.any(specular > 0.0):
                halfway = GL._normalize(light_direction + view_direction)
                specular_factor = max(0.0, float(np.dot(normal, halfway))) ** shininess
                result += specular * specular_factor * light_color

        return np.clip(result, 0.0, 1.0) * 255.0

    @staticmethod
    def _project_vertex(point):
        """Transforma um vértice 3D em coordenadas de tela e profundidade."""
        vertex = np.array([point[0], point[1], point[2], 1.0], dtype=float)
        clip = GL.projection_matrix @ GL.view_matrix @ GL.model_matrix @ vertex
        if abs(clip[3]) < 1e-12:
            return None

        ndc = clip[:3] / clip[3]
        return {
            "world": (GL.model_matrix @ vertex)[:3],
            "screen": (
                (ndc[0] + 1.0) * GL.width / 2.0,
                (1.0 - ndc[1]) * GL.height / 2.0,
            ),
            "depth": float(ndc[2]),
            "inv_w": float(1.0 / clip[3]),
        }

    @staticmethod
    def _edge(a, b, point):
        return ((point[0] - a[0]) * (b[1] - a[1])
                - (point[1] - a[1]) * (b[0] - a[0]))

    @staticmethod
    def _texture_lod(vertices, texcoords, texture):
        """Estima um nível de mipmap para a área coberta pelo triângulo."""
        if not texture or not texcoords:
            return 0

        p0, p1, p2 = [vertex["screen"] for vertex in vertices]
        uv0, uv1, uv2 = texcoords
        determinant = GL._edge(p0, p1, p2)
        if abs(determinant) < 1e-12:
            return 0

        du_dx = ((uv0[0] * (p1[1] - p2[1])
                  + uv1[0] * (p2[1] - p0[1])
                  + uv2[0] * (p0[1] - p1[1])) / determinant)
        du_dy = ((uv0[0] * (p2[0] - p1[0])
                  + uv1[0] * (p0[0] - p2[0])
                  + uv2[0] * (p1[0] - p0[0])) / determinant)
        dv_dx = ((uv0[1] * (p1[1] - p2[1])
                  + uv1[1] * (p2[1] - p0[1])
                  + uv2[1] * (p0[1] - p1[1])) / determinant)
        dv_dy = ((uv0[1] * (p2[0] - p1[0])
                  + uv1[1] * (p0[0] - p2[0])
                  + uv2[1] * (p1[0] - p0[0])) / determinant)

        # GPU.load_texture() mantém a convenção do código base, que transpõe
        # os eixos da imagem. Os eixos abaixo representam a textura original.
        width = texture[0].shape[0]
        height = texture[0].shape[1]
        footprint = max(
            abs(du_dx) * width,
            abs(du_dy) * width,
            abs(dv_dx) * height,
            abs(dv_dy) * height,
            1.0,
        )
        return min(len(texture) - 1, max(0, int(math.floor(math.log2(footprint)))))

    @staticmethod
    def _sample_texture(texture, uv):
        """Amostra uma textura com repetição e filtragem bilinear."""
        if texture is None:
            return None

        u = float(uv[0]) % 1.0
        v = float(uv[1]) % 1.0
        array_height, array_width = texture.shape[:2]
        width, height = array_height, array_width
        x = u * (width - 1)
        y = (1.0 - v) * (height - 1)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        tx, ty = x - x0, y - y0

        top = texture[x0, y0] * (1.0 - tx) + texture[x1, y0] * tx
        bottom = texture[x0, y1] * (1.0 - tx) + texture[x1, y1] * tx
        return (top * (1.0 - ty) + bottom * ty)[:3]

    @staticmethod
    def _load_texture_mipmaps(texture_name):
        """Gera os níveis de mipmap usando a rotina de textura do código base."""
        mipmaps = [gpu.GPU.load_texture(texture_name)]
        while True:
            source = mipmaps[-1]
            height, width = source.shape[:2]
            if width == 1 and height == 1:
                break

            next_height = max(1, (height + 1) // 2)
            next_width = max(1, (width + 1) // 2)
            reduced = np.empty((next_height, next_width, source.shape[2]), dtype=np.uint8)
            for y in range(next_height):
                for x in range(next_width):
                    block = source[2 * y:min(2 * y + 2, height),
                                   2 * x:min(2 * x + 2, width)]
                    reduced[y, x] = np.rint(block.mean(axis=(0, 1))).astype(np.uint8)
            mipmaps.append(reduced)
        return mipmaps

    @staticmethod
    def _write_fragment(coord, color, depth=None, alpha=1.0):
        """Escreve um fragmento usando diretamente os buffers fornecidos."""
        if not coord or alpha <= 0.0:
            return False

        framebuffer = gpu.GPU.frame_buffer[gpu.GPU.draw_framebuffer]
        x, y = int(coord[0]), int(coord[1])
        height, width = framebuffer.color.shape[:2]
        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        if depth is not None and framebuffer.depth.size != 0:
            old_depth = float(framebuffer.depth[y, x, 0])
            if depth >= old_depth:
                return False
            if alpha >= 1.0:
                framebuffer.depth[y, x, 0] = depth

        source = np.clip(np.asarray(color, dtype=float), 0.0, 255.0)
        destination = framebuffer.color[y, x, :3].astype(float)
        result = source * alpha + destination * (1.0 - alpha)
        framebuffer.color[y, x, :3] = np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)
        return True

    @staticmethod
    def _rasterize_triangle(vertices, colors, material_colors, texcoords=None,
                            texture=None, world_positions=None, normals=None):
        """Rasteriza um triângulo com interpolação, profundidade e composição."""
        points = [vertex["screen"] for vertex in vertices]
        area = GL._edge(points[0], points[1], points[2])
        if abs(area) < 1e-12:
            return

        min_x = max(0, int(math.floor(min(point[0] for point in points))))
        max_x = min(GL.width - 1, int(math.ceil(max(point[0] for point in points))))
        min_y = max(0, int(math.floor(min(point[1] for point in points))))
        max_y = min(GL.height - 1, int(math.ceil(max(point[1] for point in points))))

        base_color = GL._base_color(material_colors)
        transparency = float(material_colors.get("transparency", 0.0))
        alpha = np.clip(1.0 - transparency, 0.0, 1.0)
        mip_level = GL._texture_lod(vertices, texcoords, texture)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                sample = (x + 0.5, y + 0.5)
                weights = [
                    GL._edge(points[1], points[2], sample) / area,
                    GL._edge(points[2], points[0], sample) / area,
                    GL._edge(points[0], points[1], sample) / area,
                ]
                if min(weights) < -1e-8:
                    continue

                perspective = sum(
                    weight * vertex["inv_w"]
                    for weight, vertex in zip(weights, vertices)
                )
                if abs(perspective) < 1e-12:
                    continue
                corrected = [
                    weight * vertex["inv_w"] / perspective
                    for weight, vertex in zip(weights, vertices)
                ]

                depth = sum(
                    weight * vertex["depth"]
                    for weight, vertex in zip(weights, vertices)
                )

                world_position = None
                normal = None
                if world_positions is not None:
                    world_position = sum(
                        factor * np.asarray(position, dtype=float)
                        for factor, position in zip(corrected, world_positions)
                    )
                if normals is not None:
                    normal = GL._normalize(sum(
                        factor * np.asarray(vertex_normal, dtype=float)
                        for factor, vertex_normal in zip(corrected, normals)
                    ))

                albedo = None
                if colors is not None:
                    albedo = sum(
                        factor * np.asarray(vertex_color, dtype=float) * 255.0
                        for factor, vertex_color in zip(corrected, colors)
                    ) / 255.0

                if texcoords is not None and texture:
                    uv = sum(
                        factor * np.asarray(texcoord, dtype=float)
                        for factor, texcoord in zip(corrected, texcoords)
                    )
                    texel = GL._sample_texture(texture[mip_level], uv)
                    if texel is not None:
                        texture_color = np.clip(texel / 255.0, 0.0, 1.0)
                        albedo = texture_color if albedo is None else albedo * texture_color

                if world_position is not None and normal is not None:
                    color = GL._material_color(
                        material_colors,
                        world_position,
                        normal,
                        albedo,
                    )
                elif albedo is not None:
                    color = albedo * 255.0
                else:
                    color = base_color.copy()

                GL._write_fragment((x, y), color, depth, alpha)

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).
        lista = np.rint(GL._base_color(colors)).astype(np.uint8).tolist()
        scale = GL.supersample

        for p in range(0, len(point), 2):
            posx = int(round(point[p] * scale))
            posy = int(round(point[p + 1] * scale))
            for dx in range(scale):
                for dy in range(scale):
                    x, y = posx + dx, posy + dy
                    if 0 <= x < GL.width and 0 <= y < GL.height:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, lista)


        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        lista = np.rint(GL._base_color(colors)).astype(np.uint8).tolist()
        scale = GL.supersample

        for p in range(0, len(lineSegments) - 2, 2):
            x0 = int(round(lineSegments[p] * scale))
            y0 = int(round(lineSegments[p + 1] * scale))
            x1 = int(round(lineSegments[p + 2] * scale))
            y1 = int(round(lineSegments[p + 3] * scale))

            if y0 == y1:
                if x0 <= x1:
                    for x in range(x0, x1 + 1):
                        if 0 <= x < GL.width and 0 <= y0 < GL.height:
                            gpu.GPU.draw_pixel([x, y0],gpu.GPU.RGB8,lista)
                else:
                    for x in range(x0, x1 - 1, -1):
                        if 0 <= x < GL.width and 0 <= y0 < GL.height:
                            gpu.GPU.draw_pixel([x, y0],gpu.GPU.RGB8,lista)
            elif x0 == x1:

                if y0 <= y1:
                    for y in range(y0, y1 + 1):
                        if 0 <= x0 < GL.width and 0 <= y < GL.height:
                            gpu.GPU.draw_pixel([x0, y],gpu.GPU.RGB8,lista)
                else:
                    for y in range(y0, y1 - 1, -1):
                        if 0 <= x0 < GL.width and 0 <= y < GL.height:
                            gpu.GPU.draw_pixel([x0, y],gpu.GPU.RGB8,lista)
            else:
                dif_x = abs(x1 - x0)
                dif_y = abs(y1 - y0)
                
                if x0 < x1:
                    sx = 1
                else:
                    sx = -1

                if y0 < y1:
                    sy = 1
                else:
                    sy = -1
                erro = dif_x - dif_y

                while x0 != x1 or y0 != y1:
                    if 0 <= x0 < GL.width and 0 <= y0 < GL.height:
                        gpu.GPU.draw_pixel([x0, y0],gpu.GPU.RGB8,lista)
                    v = 2 * erro
                    if v > -dif_y:
                        erro -= dif_y
                        x0 += sx

                    if v < dif_x:
                        erro += dif_x
                        y0 += sy
                if 0 <= x1 < GL.width and 0 <= y1 < GL.height:
                    gpu.GPU.draw_pixel([x1, y1],gpu.GPU.RGB8,lista)


        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        scale = GL.supersample
        scaled_radius = radius * scale
        for deg in range(360):
            rad = math.radians(deg)
            x = round(math.cos(rad) * scaled_radius)
            y = round(math.sin(rad) * scaled_radius)

            if x < 0 or x >= GL.width or y < 0 or y >= GL.height:
                continue

            gpu.GPU.draw_pixel((x, y), gpu.GPU.RGB8,
                               np.rint(GL._base_color(colors)).astype(np.uint8).tolist())

    @staticmethod
    def _insideTriangle(triangle: Triangle, p: Vec2) -> bool:
        # Checa se o ponto p está dentro do triangulo
        def inside(
            triangle: Triangle,
            pos: Vec2
        ) -> bool:
            A, B, C = triangle
            return line(A, B, pos) >= 0 and \
                   line(B, C, pos) >= 0 and \
                   line(C, A, pos) >= 0

        # Função de linha. Retorna > 0 para dentro e < para fora.
        def line(A: Vec2, B: Vec2, X: Vec2) -> float:
            x0, y0 = A
            x1, y1 = B
            x, y = X
            return (y1 - y0) * x \
                   - (x1 - x0) * y \
                   + (x1 - x0) * y0 \
                   - (y1 - y0) * x0

        # Ajusta o ponto para pegar o meio do pixel
        def get_center_point(A: Vec2) -> Vec2:
            return (A[0] + 0.5, A[1] + 0.5)

        p = get_center_point(p)
        return inside(triangle, p)

    @staticmethod
    def newColor() -> Color:
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    @staticmethod
    def colorMultiply(color: Color) -> Color:
        return (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )

    @staticmethod
    def triangleSet2D(vertices: list[float], colors: dict[str, Color]):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        scale = GL.supersample
        for offset in range(0, len(vertices), 6):
            triangle = vertices[offset:offset + 6]
            if len(triangle) < 6:
                break
            projected = [
                {
                    "screen": (triangle[i] * scale, triangle[i + 1] * scale),
                    "depth": 0.0,
                    "inv_w": 1.0,
                }
                for i in range(0, 6, 2)
            ]
            GL._rasterize_triangle(projected, None, colors)

    @staticmethod
    def triangleSet(point, colors, vertex_colors=None):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        if not hasattr(GL, "model_matrix"):
            GL.model_matrix = np.identity(4)

        if not hasattr(GL, "view_matrix"):
            GL.viewpoint([0, 0, 10], [0, 0, 1, 0], math.pi / 4)

        projected = []
        for i in range(0, len(point), 3):
            vertex = GL._project_vertex(point[i:i + 3])
            if vertex is None:
                return
            projected.append(vertex)

        for offset in range(0, len(projected), 3):
            vertices = projected[offset:offset + 3]
            if len(vertices) < 3:
                break
            colors_for_triangle = None
            if vertex_colors:
                color_offset = offset * 3
                colors_for_triangle = [
                    vertex_colors[color_offset + i:color_offset + i + 3]
                    for i in range(0, 9, 3)
                ]
            world_positions = [vertex["world"] for vertex in vertices]
            normal = GL._normalize(np.cross(
                np.asarray(world_positions[1]) - world_positions[0],
                np.asarray(world_positions[2]) - world_positions[0],
            ))
            GL._rasterize_triangle(
                vertices,
                colors_for_triangle,
                colors,
                world_positions=world_positions,
                normals=[normal, normal, normal],
            )

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Viewpoint : ", end='')
        print("position = {0} ".format(position), end='')
        print("orientation = {0} ".format(orientation), end='')
        print("fieldOfView = {0} ".format(fieldOfView))

        T = np.identity(4)
        T[0][3] = position[0]
        T[1][3] = position[1]
        T[2][3] = position[2]

        x = orientation[0]
        y = orientation[1]
        z = orientation[2]
        angulo = orientation[3]

        tamanho = math.sqrt(x*x + y*y + z*z)

        if tamanho != 0:
            x = x / tamanho
            y = y / tamanho
            z = z / tamanho

        c = math.cos(angulo)
        s = math.sin(angulo)
        t = 1 - c

        R = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0],
            [0,           0,           0,           1]
        ])

        camera_matrix = T @ R
        GL.camera_position = np.asarray(position, dtype=float)
        GL.camera_matrix = camera_matrix
        GL.view_matrix = np.linalg.inv(camera_matrix)

        aspect = GL.width / GL.height
        f = 1 / math.tan(fieldOfView / 2)
        near = GL.near
        far = GL.far

        GL.projection_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        if not translation:
            translation = [0, 0, 0]
        if not scale:
            scale = [1, 1, 1]
        if not rotation:
            rotation = [0, 0, 1, 0]

        T = np.identity(4)
        T[0][3] = translation[0]
        T[1][3] = translation[1]
        T[2][3] = translation[2]

        S = np.identity(4)
        S[0][0] = scale[0]
        S[1][1] = scale[1]
        S[2][2] = scale[2]

        x = rotation[0]
        y = rotation[1]
        z = rotation[2]
        angulo = rotation[3]

        tamanho = math.sqrt(x*x + y*y + z*z)

        if tamanho != 0:
            x = x / tamanho
            y = y / tamanho
            z = z / tamanho

        c = math.cos(angulo)
        s = math.sin(angulo)
        t = 1 - c

        R = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0],
            [0,           0,           0,           1]
        ])

        # Guarda a transformacao do pai. A transformacao local X3D e T * R * S;
        # usando vetores-coluna, o pai precisa ficar a esquerda da matriz local.
        GL.matrix_stack.append(GL.model_matrix.copy())
        GL.model_matrix = GL.model_matrix @ T @ R @ S

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if not GL.matrix_stack:
            raise RuntimeError("transform_out chamado sem transform_in correspondente")

        GL.model_matrix = GL.matrix_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        vertex_count = len(point) // 3
        first = 0
        triangles = []

        for count in stripCount:
            count = int(count)
            if count < 3:
                first += max(count, 0)
                continue
            if first + count > vertex_count:
                raise ValueError("stripCount usa mais vertices do que point possui")

            strip = list(range(first, first + count))
            triangles.extend(GL._triangulate_strip(strip))
            first += count

        GL._draw_indexed_triangles(point, triangles, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        triangles = []
        strip = []

        # O -1 encerra apenas a tira atual; pode haver varias tiras na lista.
        for value in list(index) + [-1]:
            value = int(value)
            if value == -1:
                triangles.extend(GL._triangulate_strip(strip))
                strip = []
            elif value < -1:
                raise ValueError("indices devem ser nao negativos ou -1")
            else:
                strip.append(value)

        GL._draw_indexed_triangles(point, triangles, colors)

    @staticmethod
    def _split_indices(indexes):
        """Divide uma lista de índices X3D em faces separadas por -1."""
        faces = []
        face = []
        for value in list(indexes) + [-1]:
            value = int(value)
            if value == -1:
                if face:
                    faces.append(face)
                face = []
            else:
                face.append(value)
        return faces

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        coord_faces = GL._split_indices(coordIndex)
        color_faces = GL._split_indices(colorIndex) if colorIndex else []
        texcoord_faces = GL._split_indices(texCoordIndex) if texCoordIndex else []
        texture = None
        if current_texture:
            texture = GL._load_texture_mipmaps(current_texture[0])

        world_coordinates = [
            (GL.model_matrix @ np.array([coord[index], coord[index + 1], coord[index + 2], 1.0]))[:3]
            for index in range(0, len(coord), 3)
        ] if coord else []
        normal_sums = [np.zeros(3, dtype=float) for _ in world_coordinates]
        for face in coord_faces:
            if len(face) < 3 or any(index < 0 or index >= len(world_coordinates) for index in face):
                continue
            for corner in range(1, len(face) - 1):
                p0 = world_coordinates[face[0]]
                p1 = world_coordinates[face[corner]]
                p2 = world_coordinates[face[corner + 1]]
                face_normal = GL._normalize(np.cross(p1 - p0, p2 - p0))
                for index in (face[0], face[corner], face[corner + 1]):
                    normal_sums[index] += face_normal

        def vertex_color(index):
            if color is None or index < 0 or 3 * index + 2 >= len(color):
                return None
            return color[3 * index:3 * index + 3]

        def vertex_texcoord(index):
            if texCoord is None or index < 0 or 2 * index + 1 >= len(texCoord):
                return None
            return texCoord[2 * index:2 * index + 2]

        for face_number, face in enumerate(coord_faces):
            if len(face) < 3:
                continue
            if any(index < 0 or 3 * index + 2 >= len(coord) for index in face):
                raise ValueError("coordIndex contém um índice fora do intervalo")

            face_color_indices = color_faces[face_number] if face_number < len(color_faces) else []
            face_texcoord_indices = (
                texcoord_faces[face_number]
                if face_number < len(texcoord_faces) else []
            )

            for corner in range(1, len(face) - 1):
                corner_positions = (0, corner, corner + 1)
                triangle_indices = [face[position] for position in corner_positions]
                vertices = [
                    GL._project_vertex(coord[3 * index:3 * index + 3])
                    for index in triangle_indices
                ]
                if any(vertex is None for vertex in vertices):
                    continue

                world_positions = [vertex["world"] for vertex in vertices]
                normals = [GL._normalize(normal_sums[index]) for index in triangle_indices]
                fallback_normal = GL._normalize(np.cross(
                    np.asarray(world_positions[1]) - world_positions[0],
                    np.asarray(world_positions[2]) - world_positions[0],
                ))
                normals = [normal if np.any(normal) else fallback_normal for normal in normals]

                triangle_colors = None
                if color:
                    if colorPerVertex:
                        indices = []
                        for position, coord_index in zip(corner_positions, triangle_indices):
                            if position < len(face_color_indices):
                                indices.append(face_color_indices[position])
                            else:
                                indices.append(coord_index)
                        triangle_colors = [vertex_color(index) for index in indices]
                    else:
                        index = face_color_indices[0] if face_color_indices else face_number
                        shared = vertex_color(index)
                        triangle_colors = [shared, shared, shared]
                    if any(value is None for value in triangle_colors):
                        triangle_colors = None

                triangle_texcoords = None
                if texCoord:
                    indices = []
                    for position, coord_index in zip(corner_positions, triangle_indices):
                        if position < len(face_texcoord_indices):
                            indices.append(face_texcoord_indices[position])
                        else:
                            indices.append(coord_index)
                    triangle_texcoords = [vertex_texcoord(index) for index in indices]
                    if any(value is None for value in triangle_texcoords):
                        triangle_texcoords = None

                GL._rasterize_triangle(
                    vertices,
                    triangle_colors,
                    colors,
                    triangle_texcoords,
                    texture,
                    world_positions,
                    normals,
                )

    @staticmethod
    def _triangulate_strip(strip):
        """Converte uma tira em triangulos com orientacao consistente."""
        triangles = []
        for i in range(len(strip) - 2):
            if i % 2 == 0:
                triangles.append((strip[i], strip[i + 1], strip[i + 2]))
            else:
                triangles.append((strip[i], strip[i + 2], strip[i + 1]))
        return triangles

    @staticmethod
    def _draw_indexed_triangles(point, triangles, colors):
        """Rasteriza triângulos indexados com normais suaves por vértice."""
        vertex_count = len(point) // 3
        projected = []
        for index in range(vertex_count):
            projected.append(GL._project_vertex(point[3 * index:3 * index + 3]))

        normal_sums = [np.zeros(3, dtype=float) for _ in range(vertex_count)]
        face_normals = {}

        for triangle in triangles:
            if any(vertex_index < 0 or vertex_index >= vertex_count
                   for vertex_index in triangle):
                raise ValueError(
                    "indice de vertice fora do intervalo [0, {0})".format(vertex_count)
                )
            positions = [projected[index]["world"] for index in triangle]
            face_normal = GL._normalize(np.cross(
                np.asarray(positions[1]) - positions[0],
                np.asarray(positions[2]) - positions[0],
            ))
            face_normals[tuple(triangle)] = face_normal
            for vertex_index in triangle:
                normal_sums[vertex_index] += face_normal

        for triangle in triangles:
            vertices = [projected[index] for index in triangle]
            if any(vertex is None for vertex in vertices):
                continue
            normals = []
            for vertex_index in triangle:
                normal = GL._normalize(normal_sums[vertex_index])
                if not np.any(normal):
                    normal = face_normals[tuple(triangle)]
                normals.append(normal)
            world_positions = [vertex["world"] for vertex in vertices]
            GL._rasterize_triangle(
                vertices,
                None,
                colors,
                world_positions=world_positions,
                normals=normals,
            )

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # A cena é percorrida novamente a cada frame; por isso as luzes são
        # reconstruídas quando o NavigationInfo é encontrado.
        GL.lights = []
        if headlight:
            direction = GL.camera_matrix[:3, :3] @ np.array([0.0, 0.0, -1.0])
            GL.lights.append({
                "type": "directional",
                "ambientIntensity": 0.0,
                "color": np.ones(3, dtype=float),
                "intensity": 1.0,
                "direction": GL._normalize(direction, [0.0, 0.0, -1.0]),
            })

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        GL.lights.append({
            "type": "directional",
            "ambientIntensity": np.clip(float(ambientIntensity), 0.0, 1.0),
            "color": np.clip(np.asarray(color, dtype=float), 0.0, 1.0),
            "intensity": max(0.0, float(intensity)),
            "direction": GL._normalize(direction, [0.0, 0.0, -1.0]),
        })

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        cycle_interval = max(float(cycleInterval), 1e-12)
        key = (cycle_interval, bool(loop))
        now = time.monotonic()
        start = GL._time_sensors.setdefault(key, now)
        elapsed = max(0.0, now - start)
        if loop:
            return (elapsed % cycle_interval) / cycle_interval
        return min(1.0, elapsed / cycle_interval)

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        if not key or not keyValue or len(keyValue) < 3:
            return [0.0, 0.0, 0.0]

        values = np.asarray(keyValue, dtype=float).reshape(-1, 3)
        keys = np.asarray(key, dtype=float)
        count = min(len(keys), len(values))
        keys = keys[:count]
        values = values[:count]
        if count == 1:
            return values[0].tolist()

        fraction = float(np.clip(set_fraction, 0.0, 1.0))
        interval = int(np.searchsorted(keys, fraction, side="right") - 1)
        interval = max(0, min(interval, count - 2))
        denominator = keys[interval + 1] - keys[interval]
        t = 0.0 if abs(denominator) < 1e-12 else (
            fraction - keys[interval]
        ) / denominator

        p1 = values[interval]
        p2 = values[interval + 1]
        if closed and np.allclose(values[0], values[-1]):
            p0 = values[interval - 1] if interval > 0 else values[-2]
            p3 = values[interval + 2] if interval + 2 < count else values[1]
        else:
            p0 = values[interval - 1] if interval > 0 else p1
            p3 = values[interval + 2] if interval + 2 < count else p2

        # Catmull-Rom centripetal simplificado, com tangentes locais.
        value_changed = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
        )
        return value_changed.tolist()

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        if not key or not keyValue or len(keyValue) < 4:
            return [0.0, 0.0, 1.0, 0.0]

        keys = np.asarray(key, dtype=float)
        rotations = np.asarray(keyValue, dtype=float).reshape(-1, 4)
        count = min(len(keys), len(rotations))
        keys = keys[:count]
        rotations = rotations[:count]
        if count == 1:
            return rotations[0].tolist()

        fraction = float(np.clip(set_fraction, 0.0, 1.0))
        interval = int(np.searchsorted(keys, fraction, side="right") - 1)
        interval = max(0, min(interval, count - 2))
        denominator = keys[interval + 1] - keys[interval]
        t = 0.0 if abs(denominator) < 1e-12 else (
            fraction - keys[interval]
        ) / denominator

        def quaternion(rotation):
            axis = GL._normalize(rotation[:3], [0.0, 0.0, 1.0])
            half = float(rotation[3]) * 0.5
            return np.array([
                math.cos(half),
                axis[0] * math.sin(half),
                axis[1] * math.sin(half),
                axis[2] * math.sin(half),
            ])

        q0 = quaternion(rotations[interval])
        q1 = quaternion(rotations[interval + 1])
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = GL._normalize((1.0 - t) * q0 + t * q1)
        else:
            angle = math.acos(np.clip(dot, -1.0, 1.0))
            sine = math.sin(angle)
            result = (
                math.sin((1.0 - t) * angle) * q0
                + math.sin(t * angle) * q1
            ) / sine

        result = GL._normalize(result, [1.0, 0.0, 0.0, 0.0])
        vector = result[1:]
        vector_length = np.linalg.norm(vector)
        if vector_length < 1e-12:
            return [0.0, 0.0, 1.0, 0.0]
        angle = 2.0 * math.atan2(vector_length, result[0])
        return (vector / vector_length).tolist() + [angle]

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""

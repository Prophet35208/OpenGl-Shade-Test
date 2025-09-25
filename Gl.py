import pygame as pg
import moderngl as mgl
import sys
from camera import Camera
from light import Light
import numpy as np
import glm
import math


a = 3.0
b = 2.5
c = 4.0
segments_phi = 40
segments_theta = 40
OFFSET = 10

def createEllipsoidVertices(a, b, c, segments_phi, segments_theta):
        """
        a: Радиус по оси X.
        b: Радиус по оси Y.
        c: Радиус по оси Z.
        segments_phi: Количество сегментов по "долготе".
        segments_theta: Количество сегментов по "широте".
        """

        vertices = []
        indices = []
        normals = []

        for i in range(segments_theta + 1):
            theta = i * math.pi / segments_theta

            for j in range(segments_phi + 1):
                phi = j * 2 * math.pi / segments_phi

                x = a * math.sin(theta) * math.cos(phi)
                y = b * math.sin(theta) * math.sin(phi)
                z = c * math.cos(theta)

                vertices.extend([x, y, z])

                normalX = 2 * x / (a * a)
                normalY = 2 * y / (b * b)
                normalZ = 2 * z / (c * c)
                normal = glm.normalize(glm.vec3(normalX, normalY, normalZ))
 
                normals.extend([normal.x, normal.y, normal.z])

        for i in range(segments_theta):
            for j in range(segments_phi):
            
                p1 = i * (segments_phi + 1) + j
                p2 = i * (segments_phi + 1) + (j + 1)
                p3 = (i + 1) * (segments_phi + 1) + j
                p4 = (i + 1) * (segments_phi + 1) + (j + 1)
            
                indices.extend([p1, p2, p3])
                indices.extend([p2, p4, p3])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), np.array(normals, dtype=np.float32)

"""
Простейшая модель освещения. 
Главный параметр - нормали вершин из которых далее получаем направление света к вершине (тоже в единичном векторе)
Далее определяем интенсивность света после диффузии.
Во фрагментном шейдере закрашиваем треугольник полученным цветом (цвет в рамках одного треугольника совпадает)
"""
def createSimpleShaders(ctx):
    vertex_shader = """
        #version 330 core

        in vec3 in_position;
        in vec3 in_normal;

        uniform mat4 projection;
        uniform mat4 view;
        
        uniform vec3 light_pos;
        uniform vec3 light_color;

        flat out float intensity;

        void main() {
            gl_Position = projection * view * vec4(in_position, 1.0);

            vec3 norm = normalize(in_normal);
            vec3 frag_pos = vec3(view * vec4(in_position, 1.0));
            vec3 light_dir = normalize(light_pos - frag_pos);
            
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            intensity = length(diffuse);    
        }
    """
    fragment_shader = """
        #version 330 core

        out vec4 fragColor;
        flat in float intensity;

        void main() {
            vec3 base_color = vec3(0.5, 0.5, 0.5);
            vec3 final_color = base_color * intensity;
            fragColor = vec4(final_color, 1.0);
            }
        """
        
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    return program

def createPhongShaders(ctx):
    vertex_shader = """
            #version 330 core

            in vec3 in_position;
            in vec3 in_normal;

            uniform mat4 projection;
            uniform mat4 view;

            out vec3 normal;
            out vec3 frag_pos;
            void main() {
                gl_Position = projection * view * vec4(in_position, 1.0);
                frag_pos = vec3(view * vec4(in_position, 1.0));
                normal = in_normal;
            }
        """
    fragment_shader = """
         #version 330 core

        out vec4 fragColor;
        in vec3 normal;
        in vec3 frag_pos;
            
        uniform vec3 light_pos;

        // Интенсивности параметров света: окружающий, диффузный и зеркальный свет
        uniform vec3 ambient_intensity;  // Ia
        uniform vec3 diffuse_intensity; // Id
        uniform vec3 specular_intensity;  // Is

        void main() {
          vec3 base_color = vec3(0.5, 0.5, 0.5);

          // Вычисляем фоновую составляющую
          vec3 ambient = ambient_intensity * base_color;

          vec3 norm = normalize(normal);
          vec3 light_dir = normalize(light_pos - frag_pos);

          // Вычисляем диффузную составляющую
          float diff = max(dot(norm, light_dir), 0.0);
          vec3 diffuse = diff * diffuse_intensity * base_color;

          // Вычисляем зеркальную составляющую
          vec3 view_dir = normalize(-frag_pos);
          vec3 reflect_dir = reflect(-light_dir, norm);
          float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
          vec3 specular = spec * specular_intensity * base_color;

          // Фикс, чтобы не видеть блики на обратной стороне объекта
          if (dot(norm, view_dir) < 0.0)
            specular = vec3(0.0);
          specular *= diff;
          
          // Суммируем состовляющие света
          vec3 result = ambient + diffuse + specular;
          fragColor = vec4(result, 1.0);
        }
    """
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    return program



def createVao(ctx, program, vertices, indices, normals):
    vbo = ctx.buffer(vertices)
    ibo = ctx.buffer(indices)
    nbo = ctx.buffer(normals) 

    return ctx.vertex_array(program,
                         [(vbo, '3f', 'in_position'),
                          (nbo, '3f', 'in_normal')],
                          index_buffer=ibo)

class OpenGl:
    def __init__(self, win_size=(1600, 900)):
        # Базовая Инициализация
        pg.init()
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # Настройки мыши, чтобы не вылезала из окна
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # Создаём объект времени для анимации
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0

        self.ctx = mgl.create_context()
        self.ctx.enable(flags=mgl.DEPTH_TEST)

        # Подключам свет
        self.light = Light(position=(0,0,200))
        # Задаём камеру
        self.camera = Camera(self,position=(0,0,10))

        # Создание первого эллипса (простая модель освещения)
        vertices, indices, normals = createEllipsoidVertices(a, b, c, segments_phi, segments_theta)
        self.programSimple = createSimpleShaders(self.ctx)
        self.programSimple['projection'].write(self.camera.get_projection_matrix())
        self.programSimple['view'].write(self.camera.get_view_matrix())
        self.programSimple['light_pos'].write(self.light.position)
        self.programSimple['light_color'].write(self.light.color)
        self.vaoSimple = createVao(self.ctx, self.programSimple, vertices, indices, normals)

    
        # Второй эллипс для модели освещённости Фонга. 
        for i in range(0,len(vertices),3):
          vertices[i] += OFFSET
        
        self.programPhong = createPhongShaders(self.ctx)
        self.programPhong['projection'].write(self.camera.get_projection_matrix())
        self.programPhong['view'].write(self.camera.get_view_matrix())
        self.programPhong['light_pos'].write(self.light.position)
        self.programPhong['ambient_intensity'].write(self.light.Ia)
        self.programPhong['diffuse_intensity'].write(self.light.Id)
        self.programPhong['specular_intensity'].write(self.light.Is)
        self.vaoPhong = createVao(self.ctx, self.programPhong, vertices, indices, normals)

    
    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                # Ивент на выход
                pg.quit()
                sys.exit()
                
    def render(self):
        # Очищаем буффер, заполняеми его цветом фона
        self.ctx.clear(color=(0.08, 0.16, 0.18))

        # Рендеринг первого эллипса
        self.programSimple['projection'].write(self.camera.get_projection_matrix())
        self.programSimple['view'].write(self.camera.get_view_matrix())
        self.vaoSimple.render(mode=mgl.TRIANGLES)

        # Рендеринг второго эллипса
        self.programPhong['projection'].write(self.camera.get_projection_matrix())
        self.programPhong['view'].write(self.camera.get_view_matrix())

        self.vaoSimple.render(mode=mgl.TRIANGLES)
        self.vaoPhong.render(mode=mgl.TRIANGLES)

        # Свапаем текущий буффер на отрендеренный
        pg.display.flip()

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    # Запуск приложения
    def run(self):
        while True:
            self.get_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.delta_time = self.clock.tick(60)


if __name__ == '__main__':
    app = OpenGl()
    app.run()































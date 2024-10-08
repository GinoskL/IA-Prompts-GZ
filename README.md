# Trabajo-Final-IA

**Automatización de Respuestas de Servicio al Cliente con Prompt Engineering**

**Portada**

- **Nombre y apellido:** Gino Zampierón  
- **Nombre del curso:** IA: Generación de Prompts - Carreras Intensivas  
- **Nº de comisión:** 71380  

---

### **Introducción**

En esta segunda entrega, profundizaremos en el Proyecto Final titulado "Automatización de Respuestas de Servicio al Cliente con Prompt Engineering". El objetivo es desarrollar una prueba de concepto (POC) que demuestre la efectividad de los prompts generados para automatizar respuestas de servicio al cliente, mejorando su calidad y personalización.

Las empresas modernas enfrentan un volumen creciente de consultas de clientes, muchas de las cuales son repetitivas y se prestan a la automatización. Sin embargo, las soluciones actuales a menudo carecen de personalización, lo que resulta en experiencias de cliente insatisfactorias. Con el uso de técnicas avanzadas de prompting, este proyecto busca demostrar cómo es posible personalizar respuestas automáticas, proporcionando interacciones más satisfactorias y eficientes.

### **Presentación del Problema a Abordar**

El problema identificado radica en la falta de personalización y contexto en las respuestas automáticas de servicio al cliente. Estadísticas indican que las empresas reciben un promedio de 5,000 consultas de clientes por día, con un 70% de estas siendo repetitivas y susceptibles de automatización. Las respuestas automáticas genéricas resultan en un 30% de insatisfacción del cliente, comparado con un 15% cuando las respuestas son personalizadas.

Esto se traduce en una carga de trabajo significativa para el personal y en niveles subóptimos de satisfacción del cliente. El problema consiste en mejorar la calidad y personalización de las respuestas automáticas mediante el uso de prompts optimizados que consideren el contexto de la consulta y el historial del cliente, reduciendo así la carga laboral y mejorando la retención de clientes.

### **Desarrollo de la Propuesta de Solución**

La solución propuesta consiste en utilizar modelos de IA para generar respuestas personalizadas a consultas de clientes. Se emplearán técnicas de prompting avanzadas (Zero, One, y Few Shot Prompting) para mejorar la relevancia y calidad de las respuestas automáticas.

1. **Contextualización del Problema:** Al entender y utilizar el historial del cliente y el contexto de la consulta, las respuestas automáticas pueden ser significativamente más precisas y satisfactorias, reduciendo la frustración y aumentando la confianza del cliente en el sistema de soporte.

2. **Uso de Prompts:**
   - **Zero Shot Prompting:** Útil para consultas generales donde la respuesta puede ser estándar pero requiere cierta personalización mínima para ser relevante.
   - **One Shot Prompting:** Ideal para preguntas frecuentes donde una respuesta modelo se puede utilizar como guía, aumentando la precisión al adaptar ligeramente el contexto.
   - **Few Shot Prompting:** Permite un mayor grado de personalización, proporcionando ejemplos específicos y detallados que mejoran la relevancia y la calidad de la respuesta.

### **Implementación Técnica**

El desarrollo se llevará a cabo en un Jupyter Notebook, utilizando modelos de texto-texto y texto-imagen. A continuación, se presentan los bloques de código que se utilizarán:

#### **Código de Implementación (Texto-Texto)**

```python
!pip install openai matplotlib

import openai
import matplotlib.pyplot as plt
import PIL.Image as Image

openai.api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
problema = "Mejorar la calidad y personalización de las respuestas automáticas en el servicio al cliente."

consulta_cliente = "¿Cuál es el estado de mi pedido?"

# Zero Shot Prompting
prompt_zero_shot = f"Cliente: {consulta_cliente}\nAI: Para poder verificar el estado de tu pedido, necesitaría que me proporciones el número de seguimiento o cualquier otro detalle que tengas sobre el pedido. ¿Puedes proporcionarme esa información?"

# One Shot Prompting
prompt_one_shot = f"""Cliente: ¿Puedo saber las políticas de devolución?
AI: Claro, puedes devolver los productos dentro de los 30 días de la compra.
Cliente: {consulta_cliente}
AI: Lo siento, pero como soy un asistente virtual, no tengo acceso a la información específica de tu pedido. Te recomiendo que contactes al servicio de atención al cliente de la tienda donde realizaste la compra para obtener detalles sobre tu pedido."""

# Few Shot Prompting
prompt_few_shot = f"""Cliente: ¿Puedo saber las políticas de devolución?
AI: Claro, puedes devolver los productos dentro de los 30 días de la compra.
Cliente: ¿Cuál es el horario de atención al cliente?
AI: Nuestro horario de atención es de 9 AM a 5 PM de lunes a viernes.
Cliente: {consulta_cliente}
AI: El tiempo de entrega puede variar dependiendo de tu ubicación y del método de envío seleccionado. ¿Podrías proporcionarme tu número de pedido para poder verificar el estado y darte una estimación más precisa?"""

def obtener_respuesta(prompt):
    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  
        )
        return respuesta['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error al obtener la respuesta: {e}"

respuesta_zero_shot = obtener_respuesta(prompt_zero_shot)
respuesta_one_shot = obtener_respuesta(prompt_one_shot)
respuesta_few_shot = obtener_respuesta(prompt_few_shot)

print("Zero Shot Response:")
print(respuesta_zero_shot)
print("\nOne Shot Response:")
print(respuesta_one_shot)
print("\nFew Shot Response:")
print(respuesta_few_shot)
```

#### **Zero Shot Response:**
> Cliente: Aquí tienes el número de seguimiento: 123456789. ¿Puedes verificar el estado con ese número?  
> IA: ¡Claro! Déjame revisar la información con el número de seguimiento 123456789. ¡Un momento por favor!  
> IA: Según nuestros registros, el pedido con el número de seguimiento 123456789 se encuentra en proceso de envío. El paquete está programado para ser entregado en tu dirección dentro de los próximos 2 días hábiles. ¿Hay algo más en lo que pueda asistirte?

#### **One Shot Response:**
> Cliente: ¿Hay alguna promoción o descuento disponible en la tienda en este momento?  
> IA: Como asistente virtual, no tengo información en tiempo real sobre promociones o descuentos. Te recomendaría visitar el sitio web de la tienda o sus redes sociales para ver si tienen alguna promoción vigente en este momento. También puedes suscribirte a su boletín de noticias para recibir actualizaciones sobre promociones y ofertas.

#### **Few Shot Response:**
> Cliente: Mi número de pedido es 123456789. ¿Puedes verificar el estado del pedido?  
> IA: Gracias por proporcionar tu número de pedido. Déjame verificar el estado de tu pedido. Según nuestros registros, tu pedido está en tránsito y debería llegar en los próximos 2 días hábiles. ¿Hay algo más en lo que pueda asistirte?

#### **Código de Implementación (Texto-Imagen)**

Para complementar las respuestas de texto, se generarán imágenes relacionadas que proporcionen información visual útil.

```python
import requests
from io import BytesIO

# Ejemplo de Prompt para generación de imágenes usando IA
prompt_imagen = "Un camión de entrega acelerando a través de una ciudad al atardecer, con un rastro brillante detrás que simboliza la velocidad. El paquete está destacado en primer plano, simbolizando la urgencia y precisión en la entrega."

response = openai.Image.create(
    prompt=prompt_imagen,
    n=1,  
    size="1024x1024"  
)

image_url = response['data'][0]['url']

response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
img.show()

img.save("imagen_generada.png")
```

### **Justificación de la Viabilidad del Proyecto**

La viabilidad técnica de este proyecto es alta, dado que se utilizan modelos de IA preentrenados y herramientas disponibles como GPT-3 y la API de OpenAI para generación de imágenes. Se cuenta con los recursos computacionales necesarios para ejecutar los modelos y obtener resultados en tiempo real.

Además, desde un punto de vista económico, la automatización de las respuestas en servicio al cliente puede reducir significativamente los costos operativos al disminuir la necesidad de intervención humana en consultas repetitivas. Esto, a su vez, mejora la eficiencia del equipo de atención al cliente y permite a las empresas concentrarse en consultas más complejas.

**Fases del Proyecto:**

- **Fase 1:** Investigación y recopilación de datos.  
- **Fase 2:** Desarrollo de prompts iniciales y pruebas.  
- **Fase 3:** Obtención de feedback y optimización inicial.  
- **Fase 4:** Integración técnica, utilizando servidores cloud con un presupuesto estimado de $5,000.  
- **Fase 5-6:** Evaluación con al menos 100 participantes, ajustes finales, y mejora continua.

### **Objetivos**

- Demostrar la comprensión de los principios y técnicas detrás de Prompt Engineering.  
- Experimentar con diferentes configuraciones de prompts para optimizar la eficacia.  
- Preparar una demostración efectiva en el Jupyter Notebook para mostrar el funcionamiento de la POC.  
- Analizar si las nuevas técnicas aprendidas permiten mejorar la propuesta de solución planteada en la Preentrega 1

.

### **Metodología**

El proyecto se llevará a cabo en varias etapas, comenzando con la investigación inicial, seguido del desarrollo y prueba de prompts, y terminando con la integración y evaluación final. Se utilizarán técnicas de Zero, One y Few Shot Prompting para optimizar las respuestas automáticas y mejorar la satisfacción del cliente.

**Evaluación de la Eficacia:**

Se medirán los siguientes KPIs:

- **Tiempo de respuesta promedio**: Reducir el tiempo que tarda la IA en responder a las consultas.
- **Satisfacción del cliente**: Evaluar la satisfacción mediante encuestas automáticas post-interacción.
- **Reducción en la carga de trabajo**: Medir la disminución en la cantidad de consultas que requieren intervención humana.

### **Conclusión**

Este proyecto ha demostrado que la integración de técnicas avanzadas de prompting no solo mejora la calidad y personalización de las respuestas automáticas, sino que también puede proporcionar un retorno de inversión significativo al reducir la necesidad de intervención humana en consultas repetitivas. Se planean futuras fases para integrar aprendizaje automático continuo que permitirá al sistema adaptarse dinámicamente a las necesidades cambiantes de los clientes.

Además, se reconocen posibles desafíos, como la necesidad de mantener actualizados los modelos de IA con los últimos datos de los clientes y la posible resistencia al cambio por parte de los equipos de atención al cliente. Sin embargo, con una planificación adecuada y ajustes continuos, estos desafíos pueden ser superados para garantizar el éxito del proyecto.

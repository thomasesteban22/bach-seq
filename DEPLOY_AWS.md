# Guía de despliegue en AWS

La solución completa corre en **una sola instancia EC2**:

```
EC2 Ubuntu 24.04 (t3.large)
 ├─ mlflow.service   puerto 5000   MLflow Tracking + Model Registry + artefactos (con usuario y contraseña)
 ├─ entrenamiento    scripts/train.py ... (se corre una vez, en la misma máquina)
 └─ bach-api.service puerto 8000   FastAPI, carga el modelo champion al arrancar
```

- **Por qué en la misma máquina:** con CPU, el entrenamiento de E1–E4 tarda unos 15 minutos y no hace falta GPU. Además, la API carga el modelo desde MLflow por la red local.
- **Si el curso o un compañero ya tiene un servidor MLflow** (como en el laboratorio 2), sáltate el paso 4. En `.env` y `/etc/bach-api.env` pon la URL y las credenciales de ese servidor.

Convenciones de esta guía:

- `<IP>` es la IP elástica de la instancia.
- `<usuario-github>/bach-seq` es tu repositorio.
- `bach-chorales-grupoG` y `nlp-bach-grupoG` son los nombres del equipo. Si tu equipo tiene otro número, cámbialos en `.env` y en `/etc/bach-api.env`.

---

## 1. Subir el repositorio a GitHub (desde Windows)

1. En GitHub, crea un repositorio vacío llamado `bach-seq`, sin README ni `.gitignore`.
   - El repo puede ser **público**, porque no contiene credenciales.
   - Si lo haces **privado**, agrega al docente como colaborador (Settings → Collaborators).
2. Descomprime `bach-seq.zip`, abre CMD dentro de la carpeta `bach-seq` y ejecuta:

```cmd
git init
git add .
git commit -m "Laboratorio Bach: modelos, MLflow, API"
git branch -M main
git remote add origin https://github.com/<usuario-github>/bach-seq.git
git push -u origin main
```

Si Git pide iniciar sesión, se abre el navegador (Git Credential Manager).

Antes del commit, verifica con `git status` que **no** aparezca ningún `.env`, `.pem` ni `mlflow.db`. El `.gitignore` ya los excluye.

## 2. Crear la instancia EC2

En la consola de AWS ve a **EC2 → Launch instance**:

| Campo | Valor |
|---|---|
| Nombre | `bach-mlflow-api` |
| AMI | Ubuntu Server 24.04 LTS (x86_64) |
| Tipo | `t3.large` (2 vCPU, 8 GiB). `t3.medium` también sirve, pero entrena más lento |
| Key pair | uno tuyo (`.pem`), o `vockey` si usas AWS Academy |
| Almacenamiento | 30 GiB gp3 |

**Security group** (reglas de entrada):

| Puerto | Origen | Para qué |
|---|---|---|
| 22 (SSH) | My IP | administrar la máquina |
| 5000 (MLflow) | My IP + IP de los compañeros + IP del docente | interfaz y API de MLflow |
| 8000 (API) | My IP + IP del docente | probar la API |

La guía del laboratorio no exige publicar el servicio abierto a Internet. Si no conoces la IP del docente, abre el 8000 a `0.0.0.0/0` solo durante la sustentación y ciérralo después. MLflow queda protegido además con usuario y contraseña.

**IP elástica.** Ve a EC2 → Elastic IPs → Allocate y luego **Associate** a la instancia. Así la IP no cambia cuando la instancia se apaga y se vuelve a prender, y las URLs que entregas siguen sirviendo.

> En AWS Academy (Learner Lab) la instancia se apaga cuando termina la sesión del laboratorio. Los servicios de esta guía quedan configurados para arrancar solos cuando la instancia vuelve a encender.

## 3. Preparar la máquina

Conéctate por SSH:

```bash
ssh -i <llave>.pem ubuntu@<IP>
```

Ya en la EC2, ejecuta:

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/<usuario-github>/bach-seq.git
cd ~/bach-seq
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# torch solo para CPU (unos 200 MB en lugar de ~3 GB con CUDA)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Si el repositorio es privado, `git clone` pide usuario y contraseña. En la contraseña pega un **token personal de GitHub** (Settings → Developer settings → Fine-grained token, con permiso de solo lectura sobre ese repo). No lo guardes en la máquina.

## 4. Servidor MLflow

```bash
mkdir -p ~/mlflow-server
cp ~/bach-seq/deploy/basic_auth.ini ~/mlflow-server/
chmod 600 ~/mlflow-server/basic_auth.ini
sudo cp ~/bach-seq/deploy/mlflow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mlflow
curl -s localhost:5000/health        # debe responder OK
```

**Cambia de inmediato la contraseña inicial del admin**, que es `password`, y crea un usuario de solo lectura para el docente. Escribe las contraseñas reales en lugar de los `<...>`:

```bash
curl -u admin:password -X PATCH localhost:5000/api/2.0/mlflow/users/update-password \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<contrasena-admin>"}'

curl -u admin:<contrasena-admin> -X POST localhost:5000/api/2.0/mlflow/users/create \
  -H "Content-Type: application/json" \
  -d '{"username":"docente","password":"<contrasena-docente>"}'
```

El usuario `docente` puede **ver** todo, pero no puede borrar ni modificar nada, porque `default_permission = READ`. Comparte su contraseña con el docente por un canal privado, nunca en el repo.

Para borrar esos comandos del historial de la terminal, ejecuta `history -c`.

Comprueba en el navegador que `http://<IP>:5000` pida usuario y contraseña.

## 5. Entrenar, seleccionar, generar y registrar (en la EC2)

Crea el archivo de credenciales. Queda solo en la máquina, porque `.gitignore` lo excluye.

```bash
cd ~/bach-seq
cp .env.example .env
nano .env
```

El contenido de `.env` debe quedar así:

```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=<contrasena-admin>
MLFLOW_EXPERIMENT=nlp-bach-grupoG
MODEL_NAME=bach-chorales-grupoG
```

Luego:

```bash
chmod 600 .env
set -a && source .env && set +a
source .venv/bin/activate
```

Corre el pipeline completo dentro de `tmux`, para que no se corte si se cae el SSH:

```bash
tmux new -s train
python scripts/train.py 2>&1 | tee train.log          # E1, E2, E3 y E4 (~15 min)
python scripts/select_and_test.py                      # elige por val_loss y evalúa test una vez
python scripts/generate_midi.py                        # 2 MIDI + mapa de atención de E3
python scripts/register_model.py                       # Model Registry + alias champion
(cd /tmp && python ~/bach-seq/scripts/check_load.py)   # carga desde otro proceso
```

- `Ctrl+B` y luego `D` te sacan de tmux sin detener el proceso. `tmux attach -t train` te devuelve a él.
- El último comando debe imprimir algo como `Cargado bach-chorales-grupoG v1 (run ...) ...; generado (33, 4)`.

Para hacer otra vez el E4 con un cambio distinto, puedes correr por ejemplo `python scripts/train.py --runs E4 --e4-param lr --e4-value 0.002`. Después repite la selección, la generación y el registro.

## 6. API como servicio

```bash
sudo cp ~/bach-seq/deploy/bach-api.env.example /etc/bach-api.env
sudo nano /etc/bach-api.env          # pon la contraseña real del admin
sudo chmod 600 /etc/bach-api.env
sudo cp ~/bach-seq/deploy/bach-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bach-api
journalctl -u bach-api -n 20         # debe decir: Modelo cargado: bach-chorales-grupoG v1
```

Pruebas desde la propia EC2:

```bash
curl -s localhost:8000/health
python scripts/api_smoke.py http://localhost:8000 | tee results/api_smoke.txt
pytest -v tests/test_api.py -p no:warnings | tee results/pytest_api.txt
```

Desde tu PC:

- abre `http://<IP>:8000/docs`, o
- ejecuta `python scripts/api_smoke.py http://<IP>:8000`.

Si cambias el alias `champion` a otra versión, ejecuta `sudo systemctl restart bach-api`. La API solo consulta MLflow al arrancar.

## 7. Llevar los resultados reales al repositorio

Los números del README salieron de una corrida de prueba. Tienes que reemplazarlos por los de la EC2. Desde CMD en Windows, dentro de la carpeta del repo:

```cmd
scp -i <llave>.pem -r ubuntu@<IP>:~/bach-seq/results .
type results\comparison.csv
type results\selection.json
```

Con esos archivos actualiza:

- la tabla E1–E4 (sección 5 del README);
- el resultado de test (sección 6);
- las estadísticas de generación (`results/generation/generation_report.json`) y la lectura del mapa de atención (sección 7);
- la versión registrada (sección 8).

Revisa también que el texto del análisis siga cuadrando con los números nuevos. Luego:

```cmd
git add results README.md
git commit -m "Resultados de la corrida en AWS"
git push
```

En la EC2 corre `cd ~/bach-seq && git pull` para que el repo de la máquina quede igual.

## 8. Qué entregar al docente

| Entregable | Valor |
|---|---|
| Repositorio | `https://github.com/<usuario-github>/bach-seq` |
| Experimento MLflow | `http://<IP>:5000/#/experiments/<id>`, con usuario `docente` y su contraseña (por canal privado) |
| Modelo registrado | `bach-chorales-grupoG`, alias `champion`, versión y run de origen (en MLflow → Models) |
| API | `http://<IP>:8000/docs` (`/health` y `/generate`) |

## 9. Antes de la sustentación

1. Prende la instancia y verifica que los dos servicios estén activos:
   ```bash
   systemctl status mlflow bach-api
   ```
2. Ejecuta `curl http://<IP>:8000/health` desde tu PC.
3. Si el docente se conecta desde otra red, agrega su IP al security group en los puertos 5000 y 8000.

## Problemas comunes

| Síntoma | Causa y solución |
|---|---|
| `pip` descarga paquetes `nvidia-*` de varios GB | Faltó instalar torch CPU **antes** de `requirements.txt` (paso 3). |
| `401 Unauthorized` al entrenar o en la API | Falta cargar `.env` (`set -a && source .env && set +a`) o la contraseña en `/etc/bach-api.env` está mal. |
| `bach-api` se reinicia en bucle | Revisa `journalctl -u bach-api -n 50`. Lo usual es que todavía no exista la versión `champion` (falta el paso 5) o que las credenciales estén mal. |
| No abre `http://<IP>:8000` desde tu PC | El security group no incluye tu IP actual, o la IP de la instancia cambió (asocia la IP elástica). |
| `ImportError ... FallbackAsyncAdaptedQueuePool` | Hay una versión de SQLAlchemy incompatible. `requirements.txt` fija `sqlalchemy==2.0.36`; reinstala con `pip install -r requirements.txt`. |

# config/m365/utils/temp_utils.py
import os
import tempfile
import atexit
import signal
import platform
import weakref
import gc
import sys
import threading
import time
import uuid

class TempManager:
    """
    Manages a temporary directory that gets deleted reliably across different environments
    and maintains isolation between different Jupyter notebooks or Python processes.
    
    This class provides a robust solution for creating and cleaning up temporary directories
    that works reliably in different operating systems and execution environments including
    regular Python scripts, Jupyter notebooks, and interactive shells.
    
    Key features:
    - Process isolation: Each instance is tied to a specific process ID
    - Multiple cleanup mechanisms: Uses several approaches to ensure cleanup happens
    - Cross-platform support: Works on Windows, macOS and Linux
    - Jupyter-aware: Special handling for Jupyter notebook environments
    """
    
    # Diccionario para mantener instancias por ID de proceso
    _instances_by_pid = {}
    
    def __init__(self):
        """
        Initialize a new TempManager instance.
        
        Creates a temporary directory with a unique identifier and registers
        multiple cleanup methods to ensure the directory gets deleted reliably.
        """
        # Crear un ID único para esta instancia que incluye el PID para aislamiento
        self.instance_id = f"{os.getpid()}-{uuid.uuid4()}"
        
        # Crear directorio temporal
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_name = self.temp_dir.name
        print(f"📂 Temporary folder created: {self.dir_name} (Instance: {self.instance_id})")
        
        # Registrar múltiples métodos de limpieza
        self._register_cleanup_methods()
        
        # Registrar esta instancia en el diccionario de instancias por PID
        pid = os.getpid()
        if pid not in self.__class__._instances_by_pid:
            self.__class__._instances_by_pid[pid] = set()
        self.__class__._instances_by_pid[pid].add(weakref.ref(self))
    
    def _register_cleanup_methods(self):
        """
        Registers multiple cleanup methods for greater reliability.
        
        This method sets up several different mechanisms to ensure the temporary
        directory gets cleaned up under various scenarios:
        1. Standard atexit handler for normal program termination
        2. Signal handlers for abnormal termination (SIGINT, SIGTERM, etc.)
        3. Python finalizer for object garbage collection
        4. Background monitoring thread (especially useful in Windows)
        5. Jupyter-specific cleanup for notebook environments
        """
        # 1. Método estándar: atexit
        atexit.register(self.cleanup)
        
        # 2. Manejo de señales para terminación anormal
        self._setup_signal_handlers()
        
        # 3. Método específico para Windows: Finalizer de Python
        weakref.finalize(self, self._class_cleanup, self.dir_name, self.instance_id)
        
        # 4. Monitor de proceso en segundo plano (útil en entornos como Jupyter)
        if platform.system() == "Windows":
            self._start_monitoring_thread()
        
        # 5. Limpieza en Jupyter Notebook (si es necesario)
        self._register_jupyter_cleanup()
    
    @classmethod
    def _class_cleanup(cls, dir_name, instance_id):
        """
        Static method to clean up a known directory.
        
        This method is designed to be called by the weakref finalizer and directly
        attempts to remove the specified directory without relying on instance state.
        
        Args:
            dir_name (str): Path to the directory that needs cleaning
            instance_id (str): Identifier for logging purposes
        """
        try:
            print(f"🧹 Finalizer cleaning up: {dir_name} (Instance: {instance_id})")
            if os.path.exists(dir_name):
                import shutil
                shutil.rmtree(dir_name, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ Error in finalizer cleanup: {str(e)}")
    
    def _setup_signal_handlers(self):
        """
        Sets up signal handlers compatible with the current operating system.
        
        Configures handlers for common termination signals (SIGINT, SIGTERM) on all
        platforms, with additional signals (SIGHUP, SIGQUIT) on Unix-like systems.
        The handlers ensure cleanup only for instances in the current process.
        """
        # Señales básicas disponibles en todos los sistemas
        signals_to_handle = []
        for sig_name in ['SIGINT', 'SIGTERM']:
            if hasattr(signal, sig_name):
                signals_to_handle.append(getattr(signal, sig_name))
        
        # Señales adicionales en sistemas Unix
        if platform.system() != "Windows":
            for sig_name in ['SIGHUP', 'SIGQUIT']:
                if hasattr(signal, sig_name):
                    signals_to_handle.append(getattr(signal, sig_name))
        
        # Definir manejador que solo limpia instancias del proceso actual
        pid = os.getpid()
        def signal_handler(sig, frame):
            print(f"🛑 Process {pid} received signal {sig}, cleaning up...")
            # Solo limpiar instancias de este proceso
            self.__class__.cleanup_by_pid(pid)
            sys.exit(0)
        
        # Registrar señales
        for sig in signals_to_handle:
            try:
                # Usar un manejador personalizado para este proceso
                prev_handler = signal.getsignal(sig)
                if prev_handler not in (signal.SIG_IGN, signal.SIG_DFL, None):
                    # Ya existe un manejador, no sobrescribir
                    continue
                signal.signal(sig, signal_handler)
            except Exception:
                pass
    
    def _start_monitoring_thread(self):
        """
        Starts a thread that monitors if this specific instance is being cleaned up.
        
        This is especially useful in Windows environments where signal handling may
        be less reliable. The thread periodically checks if the instance is still valid
        and attempts cleanup if something goes wrong.
        """
        def monitor_thread():
            instance_id = self.instance_id  # Capturar el ID para referencia
            while True:
                try:
                    time.sleep(1)
                    # Verificar si esta instancia específica aún existe
                    if not hasattr(self, 'temp_dir') or self.temp_dir is None:
                        break
                except Exception:
                    # Si ocurre cualquier error, intentar limpiar solo esta instancia
                    self.cleanup()
                    break
        
        # Crear y comenzar el hilo como daemon
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
    
    def _register_jupyter_cleanup(self):
        """
        Registers a cleanup handler specifically for Jupyter Notebook environments.
        
        Detects if running in a Jupyter environment and sets up appropriate cleanup
        mechanisms when the kernel shuts down.
        """
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                # Estamos en un entorno de Jupyter
                def cleanup_on_kernel_shutdown():
                    print("🔄 Jupyter kernel shutdown detected. Cleaning up temporary files...")
                    self.__class__.cleanup_by_pid(os.getpid())
                
                # Registrar limpieza al cerrar el kernel (usando atexit como respaldo)
                atexit.register(cleanup_on_kernel_shutdown)
                # print("✅ Jupyter kernel shutdown cleanup handler registered.")
        except ImportError:
            # IPython no está instalado, no es un entorno de Jupyter
            pass
    
    @classmethod
    def cleanup_by_pid(cls, pid):
        """
        Cleans up all instances associated with a specific process ID.
        
        This class method provides a way to clean up all temporary directories
        created by a particular process, which is useful when a process is terminating.
        
        Args:
            pid (int): Process ID whose instances should be cleaned up
        """
        if pid not in cls._instances_by_pid:
            return
            
        for ref in list(cls._instances_by_pid[pid]):
            instance = ref()
            if instance is not None:
                instance.cleanup()
            cls._instances_by_pid[pid].discard(ref)
            
        # Si ya no hay instancias, eliminar la entrada del PID
        if not cls._instances_by_pid[pid]:
            del cls._instances_by_pid[pid]
    
    @classmethod
    def cleanup_all(cls):
        """
        Cleans up all instances across all processes.
        
        This class method provides a way to clean up all temporary directories
        created by all processes. Use with caution as it affects all known instances.
        """
        for pid in list(cls._instances_by_pid.keys()):
            cls.cleanup_by_pid(pid)
    
    def get_temp_path(self, filename=None):
        """
        Returns the path to the temp directory or a specific file inside it.
        
        This method provides a convenient way to get the path to the temporary
        directory or construct a path to a file within it.
        
        Args:
            filename (str, optional): If provided, returns path to this file within
                                     the temporary directory. Defaults to None.
        
        Returns:
            str: Path to the temporary directory or specific file
            
        Raises:
            RuntimeError: If the temporary directory has already been cleaned up
        """
        if not hasattr(self, 'temp_dir') or self.temp_dir is None:
            raise RuntimeError(f"Temporary directory for instance {self.instance_id} has already been cleaned up")
        
        if filename:
            # Asegurar que el nombre de archivo es seguro
            safe_filename = os.path.basename(str(filename))
            return os.path.join(self.dir_name, safe_filename)
        return self.dir_name
    
    def cleanup(self):
        """
        Manually deletes the temporary directory.
        
        This method allows for explicit cleanup of the temporary directory before
        automatic cleanup mechanisms are triggered. It includes fallback cleanup
        attempts in case the standard method fails.
        """
        if hasattr(self, 'temp_dir') and self.temp_dir is not None:
            try:
                temp_dir = self.temp_dir
                dir_name = self.dir_name
                instance_id = self.instance_id
                # Marcar como limpiado antes de intentar limpiar
                self.temp_dir = None
                # Intentar la limpieza normal
                temp_dir.cleanup()
                print(f"🗑️ Temporary folder deleted: {dir_name} (Instance: {instance_id})")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {str(e)}")
                # Intento de limpieza de emergencia
                try:
                    if os.path.exists(self.dir_name):
                        import shutil
                        shutil.rmtree(self.dir_name, ignore_errors=True)
                except Exception:
                    pass

# Función de fábrica para crear instancias únicas
def create_temp_manager():
    """
    Creates a new isolated instance of TempManager.
    
    Factory function that returns a fresh TempManager instance.
    This is useful when multiple independent temporary directories are needed.
    
    Returns:
        TempManager: A new TempManager instance
    """
    return TempManager()

# Crea la instancia global pero permite crear instancias adicionales si es necesario
temp_manager = create_temp_manager()

# Registrar limpieza al salir, pero solo para el proceso actual
atexit.register(TempManager.cleanup_by_pid, os.getpid())

# Función para limpiar al descargar el módulo
def _cleanup_module():
    """
    Function to clean up only instances from the current process when unloading the module.
    
    This is a cleanup handler that gets called when the module is unloaded,
    ensuring temporary directories are cleaned even if normal cleanup fails.
    """
    TempManager.cleanup_by_pid(os.getpid())

# Registrar limpieza al descargar el módulo
sys.modules[__name__].__del__ = _cleanup_module
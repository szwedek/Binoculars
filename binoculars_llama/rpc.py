import ctypes
from llama_cpp import Llama
from llama_cpp.llama_cpp import _lib

def add_rpc_devices(servers: str) -> None:
    lib = _lib
    
    lib.ggml_backend_reg_by_name.argtypes = [ctypes.c_char_p]
    lib.ggml_backend_reg_by_name.restype = ctypes.c_void_p
    rpc_reg = lib.ggml_backend_reg_by_name(b"RPC")
    
    if not rpc_reg:
        raise ValueError("Failed to find backend registry with name 'RPC'.")
    
    lib.ggml_backend_reg_get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ggml_backend_reg_get_proc_address.restype = ctypes.c_void_p
    rpc_add_fn_ptr = lib.ggml_backend_reg_get_proc_address(rpc_reg, b"ggml_backend_rpc_add_device")
    
    if not rpc_add_fn_ptr:
        raise ValueError("Failed to find the function for adding RPC devices.")
    
    PROTOTYPE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
    rpc_add_fn = PROTOTYPE(rpc_add_fn_ptr)
    
    lib.ggml_backend_device_register.argtypes = [ctypes.c_void_p]
    lib.ggml_backend_device_register.restype = None
    
    for server in servers.split(','):
        server = server.strip().encode("utf-8")
        dev = rpc_add_fn(server)
        if dev:
            lib.ggml_backend_device_register(dev)
            print(f"Registered RPC device: {server.decode('utf-8')}")
        else:
            raise ValueError(f"Failed to register RPC device for server: {server.decode('utf-8')}")
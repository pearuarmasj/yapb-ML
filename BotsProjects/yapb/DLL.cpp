#include <windows.h>

// Simple payload: just runs a thread on attach
DWORD WINAPI MainThread(LPVOID lpParam) {
    // Replace with whatever you want (input, cheats, movement, etc)
    MessageBoxA(NULL, "Hello from injected DLL!", "Injected", MB_OK);

    // Example: Write to engine memory, call engine functions, whatever

    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
        CreateThread(NULL, 0, MainThread, NULL, 0, NULL);
    }
    return TRUE;
}
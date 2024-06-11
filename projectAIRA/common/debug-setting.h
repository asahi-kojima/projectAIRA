#pragma once

#ifdef _DEBUG
#define DEBUG_MODE 1
#else
#define DEBUG_MODE 0
#endif

#define CPU_DEBUG_ON (0 & DEBUG_MODE)
#define INDEX_DEBUG (1 & DEBUG_MODE)
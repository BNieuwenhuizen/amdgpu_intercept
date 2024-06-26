extern "C" {
#include "amdgpu.h"
#include "amdgpu_drm.h"
#include "xf86drm.h"
}

#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <vector>

enum radeon_class {
                 GFX6, GFX7, GFX8, GFX9, GFX10, GFX11
};
enum radeon_class chip_class = GFX11;

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#include "sid.h"
#include "sid_tables.h"

#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_CYAN "\033[1;36m"

#define INDENT_PKT 8

static std::mutex global_mutex;
static std::string internal_output_dir;
static bool internal_dump_draws = false;

static std::once_flag env_vars_once_flag;
static void parse_env_vars() {
  auto dir = getenv("INTERCEPT_DIR");
  if (!dir || dir[0] == 0)
    internal_output_dir = "/tmp/";
  else {
    internal_output_dir = dir;
    if (internal_output_dir.back() != '/')
      internal_output_dir += '/';
  }

  auto dump_env = getenv("DUMP_DRAWS");
  internal_dump_draws = false;
  if (dump_env)
    internal_dump_draws = atoi(dump_env);
}


static bool get_dump_draws() {
  std::call_once(env_vars_once_flag, parse_env_vars);
  return internal_dump_draws;
}

static const std::string &get_output_dir() {
  std::call_once(env_vars_once_flag, parse_env_vars);
  return internal_output_dir;
}

static void* load_libdrm_handle() {
  void *handle = dlopen("/usr/lib/libdrm_amdgpu.so", RTLD_LOCAL | RTLD_LAZY);
  if (handle)
    return handle;
  fprintf(stderr, "internal libdrm_amdgpu.so not found\n");
  abort();
}

struct VTable {
  typeof(&amdgpu_cs_ctx_create2)               fn_cs_ctx_create2;
  typeof(&amdgpu_cs_syncobj_import_sync_file)  fn_cs_syncobj_import_sync_file;
  typeof(&amdgpu_cs_query_reset_state)         fn_cs_query_reset_state;
  typeof(&amdgpu_bo_cpu_unmap)                 fn_bo_cpu_unmap;
  typeof(&amdgpu_va_range_alloc)               fn_va_range_alloc;
  typeof(&amdgpu_cs_syncobj_import_sync_file2) fn_cs_syncobj_import_sync_file2;
  typeof(&amdgpu_cs_signal_semaphore)          fn_cs_signal_semaphore;
  typeof(&amdgpu_create_bo_from_user_mem)      fn_create_bo_from_user_mem;
  typeof(&amdgpu_cs_syncobj_export_sync_file2) fn_cs_syncobj_export_sync_file2;
  typeof(&amdgpu_bo_list_destroy_raw)          fn_bo_list_destroy_raw;
  typeof(&amdgpu_cs_syncobj_reset)             fn_cs_syncobj_reset;
  typeof(&amdgpu_query_crtc_from_id)           fn_query_crtc_from_id;
  typeof(&amdgpu_bo_cpu_map)                   fn_bo_cpu_map;
  typeof(&amdgpu_vm_reserve_vmid)              fn_vm_reserve_vmid;
  typeof(&amdgpu_query_hw_ip_info)             fn_query_hw_ip_info;
  typeof(&amdgpu_bo_list_destroy)              fn_bo_list_destroy;
  typeof(&amdgpu_query_buffer_size_alignment)  fn_query_buffer_size_alignment;
  typeof(&amdgpu_bo_wait_for_idle)             fn_bo_wait_for_idle;
  typeof(&amdgpu_cs_syncobj_wait)              fn_cs_syncobj_wait;
  typeof(&amdgpu_cs_ctx_override_priority)     fn_cs_ctx_override_priority;
  typeof(&amdgpu_query_hw_ip_count)            fn_query_hw_ip_count;
  typeof(&amdgpu_query_info)                   fn_query_info;
  typeof(&amdgpu_query_sw_info)                fn_query_sw_info;
  typeof(&amdgpu_cs_syncobj_transfer)          fn_cs_syncobj_transfer;
  typeof(&amdgpu_query_gpu_info)               fn_query_gpu_info;
  typeof(&amdgpu_cs_chunk_fence_to_dep)        fn_cs_chunk_fence_to_dep;
  typeof(&amdgpu_cs_destroy_semaphore)         fn_cs_destroy_semaphore;
  typeof(&amdgpu_cs_ctx_create)                fn_cs_ctx_create;
  typeof(&amdgpu_cs_syncobj_timeline_wait)     fn_cs_syncobj_timeline_wait;
  typeof(&amdgpu_read_mm_registers)            fn_read_mm_registers;
  typeof(&amdgpu_cs_import_syncobj)            fn_cs_import_syncobj;
  typeof(&amdgpu_query_sensor_info)            fn_query_sensor_info;
  typeof(&amdgpu_bo_query_info)                fn_bo_query_info;
  typeof(&amdgpu_bo_set_metadata)              fn_bo_set_metadata;
  typeof(&amdgpu_bo_list_create)               fn_bo_list_create;
  typeof(&amdgpu_bo_va_op_raw)                 fn_bo_va_op_raw;
  typeof(&amdgpu_device_initialize)            fn_device_initialize;
  typeof(&amdgpu_cs_wait_fences)               fn_cs_wait_fences;
  typeof(&amdgpu_vm_unreserve_vmid)            fn_vm_unreserve_vmid;
  typeof(&amdgpu_bo_export)                    fn_bo_export;
  typeof(&amdgpu_device_deinitialize)          fn_device_deinitialize;
  typeof(&amdgpu_get_marketing_name)           fn_get_marketing_name;
  typeof(&amdgpu_cs_syncobj_export_sync_file)  fn_cs_syncobj_export_sync_file;
  typeof(&amdgpu_cs_chunk_fence_info_to_data)  fn_cs_chunk_fence_info_to_data;
  typeof(&amdgpu_cs_create_syncobj)            fn_cs_create_syncobj;
  typeof(&amdgpu_va_range_query)               fn_va_range_query;
  typeof(&amdgpu_cs_query_reset_state2)        fn_cs_query_reset_state2;
  typeof(&amdgpu_bo_alloc)                     fn_bo_alloc;
  typeof(&amdgpu_va_range_free)                fn_va_range_free;
  typeof(&amdgpu_cs_export_syncobj)            fn_cs_export_syncobj;
  typeof(&amdgpu_bo_va_op)                     fn_bo_va_op;
  typeof(&amdgpu_bo_inc_ref)                   fn_bo_inc_ref;
  typeof(&amdgpu_find_bo_by_cpu_mapping)       fn_find_bo_by_cpu_mapping;
  typeof(&amdgpu_cs_submit)                    fn_cs_submit;
  typeof(&amdgpu_query_gds_info)               fn_query_gds_info;
  typeof(&amdgpu_cs_submit_raw2)               fn_cs_submit_raw2;
  typeof(&amdgpu_cs_destroy_syncobj)           fn_cs_destroy_syncobj;
  typeof(&amdgpu_query_heap_info)              fn_query_heap_info;
  typeof(&amdgpu_cs_create_syncobj2)           fn_cs_create_syncobj2;
  typeof(&amdgpu_cs_submit_raw)                fn_cs_submit_raw;
  typeof(&amdgpu_cs_create_semaphore)          fn_cs_create_semaphore;
  typeof(&amdgpu_cs_wait_semaphore)            fn_cs_wait_semaphore;
  typeof(&amdgpu_bo_list_update)               fn_bo_list_update;
  typeof(&amdgpu_bo_import)                    fn_bo_import;
  typeof(&amdgpu_cs_syncobj_timeline_signal)   fn_cs_syncobj_timeline_signal;
  typeof(&amdgpu_cs_fence_to_handle)           fn_cs_fence_to_handle;
  typeof(&amdgpu_cs_ctx_free)                  fn_cs_ctx_free;
  typeof(&amdgpu_cs_query_fence_status)        fn_cs_query_fence_status;
  typeof(&amdgpu_cs_syncobj_signal)            fn_cs_syncobj_signal;
  typeof(&amdgpu_cs_syncobj_query2)            fn_cs_syncobj_query2;
  typeof(&amdgpu_cs_syncobj_query)             fn_cs_syncobj_query;
  typeof(&amdgpu_query_firmware_version)       fn_query_firmware_version;
  typeof(&amdgpu_bo_list_create_raw)           fn_bo_list_create_raw;
  typeof(&amdgpu_bo_free)                      fn_bo_free;
};

static VTable vtable;
static std::once_flag vtable_once_flag;

static void load_vtable() {
  void *handle = load_libdrm_handle();
#define HANDLE(v)                                                  \
  vtable.fn_##v = (typeof(&amdgpu_##v))dlsym(handle, "amdgpu_" #v); \
  if (!vtable.fn_##v) {                                            \
    fprintf(stderr, "Failed to load \"amdgpu_" #v "\"\n");         \
    abort();                                                       \
  }
  HANDLE(cs_ctx_create2)
  HANDLE(cs_syncobj_import_sync_file)
  HANDLE(cs_query_reset_state)
  HANDLE(bo_cpu_unmap)
  HANDLE(va_range_alloc)
  HANDLE(cs_syncobj_import_sync_file2)
  HANDLE(cs_signal_semaphore)
  HANDLE(create_bo_from_user_mem)
  HANDLE(cs_syncobj_export_sync_file2)
  HANDLE(bo_list_destroy_raw)
  HANDLE(cs_syncobj_reset)
  HANDLE(query_crtc_from_id)
  HANDLE(bo_cpu_map)
  HANDLE(vm_reserve_vmid)
  HANDLE(query_hw_ip_info)
  HANDLE(bo_list_destroy)
  HANDLE(query_buffer_size_alignment)
  HANDLE(bo_wait_for_idle)
  HANDLE(cs_syncobj_wait)
  HANDLE(cs_ctx_override_priority)
  HANDLE(query_hw_ip_count)
  HANDLE(query_info)
  HANDLE(query_sw_info)
  HANDLE(cs_syncobj_transfer)
  HANDLE(query_gpu_info)
  HANDLE(cs_chunk_fence_to_dep)
  HANDLE(cs_destroy_semaphore)
  HANDLE(cs_ctx_create)
  HANDLE(cs_syncobj_timeline_wait)
  HANDLE(read_mm_registers)
  HANDLE(cs_import_syncobj)
  HANDLE(query_sensor_info)
  HANDLE(bo_query_info)
  HANDLE(bo_set_metadata)
  HANDLE(bo_list_create)
  HANDLE(bo_va_op_raw)
  HANDLE(device_initialize)
  HANDLE(cs_wait_fences)
  HANDLE(vm_unreserve_vmid)
  HANDLE(bo_export)
  HANDLE(device_deinitialize)
  HANDLE(get_marketing_name)
  HANDLE(cs_syncobj_export_sync_file)
  HANDLE(cs_chunk_fence_info_to_data)
  HANDLE(cs_create_syncobj)
  HANDLE(va_range_query)
  HANDLE(cs_query_reset_state2)
  HANDLE(bo_alloc)
  HANDLE(va_range_free)
  HANDLE(cs_export_syncobj)
  HANDLE(bo_va_op)
  HANDLE(bo_inc_ref)
  HANDLE(find_bo_by_cpu_mapping)
  HANDLE(cs_submit)
  HANDLE(query_gds_info)
  HANDLE(cs_submit_raw2)
  HANDLE(cs_destroy_syncobj)
  HANDLE(query_heap_info)
  HANDLE(cs_create_syncobj2)
  HANDLE(cs_submit_raw)
  HANDLE(cs_create_semaphore)
  HANDLE(cs_wait_semaphore)
  HANDLE(bo_list_update)
  HANDLE(bo_import)
  HANDLE(cs_syncobj_timeline_signal)
  HANDLE(cs_fence_to_handle)
  HANDLE(cs_ctx_free)
  HANDLE(cs_query_fence_status)
  HANDLE(cs_syncobj_signal)
  HANDLE(cs_syncobj_query2)
  HANDLE(cs_syncobj_query)
  HANDLE(query_firmware_version)
  HANDLE(bo_list_create_raw)
  HANDLE(bo_free)
#undef HANDLE
}

const VTable& get_vtable() {
  std::call_once(vtable_once_flag, load_vtable);
  return vtable;
}

struct Buffer_info {
  void *data = nullptr;
  struct amdgpu_bo_alloc_request request;
  
};

struct Map_info {
  amdgpu_bo_handle bo;
  std::uint64_t addr;
  std::uint64_t size;
  std::uint64_t offset;
};

static std::map<amdgpu_bo_handle, Buffer_info> buffers;
static std::map<std::uint64_t, Map_info> maps;
static std::map<amdgpu_bo_list_handle, std::vector<std::pair<amdgpu_bo_handle, int>>> bo_lists;

void *get_ptr(uint64_t addr, uint64_t size) {
  std::unique_lock<std::mutex> lock(global_mutex);
  auto it = maps.upper_bound(addr);
  if (it == maps.begin()) {
    fprintf(stderr, "map not found %llx\n", addr);
    return nullptr;
  }
  --it;
  if (it == maps.end()) {
    fprintf(stderr, "map not found %llx\n", addr);
    return nullptr;
  }
  if (it->second.addr > addr ||
      it->second.addr + it->second.size < addr + size) {
    fprintf(stderr, "map too small\n");
    return nullptr;
  }
  auto buf_it = buffers.find(it->second.bo);
  if (buf_it == buffers.end()) {
    fprintf(stderr, "could not find buffer associated with map\n");
    return nullptr;
  }
  if (!buf_it->second.data) {
    lock.unlock();
    int ret = amdgpu_bo_cpu_map(buf_it->first, &buf_it->second.data);
    if (ret) {
      fprintf(stderr, "failed to cpu map, %d\n", ret);
      return nullptr;
    }
    lock.lock();
  }

  return (char *)buf_it->second.data + it->second.offset +
         (addr - it->second.addr);
}

static void print_spaces(std::ostream &os, unsigned num) {
  for (unsigned i = 0; i < num; ++i)
    os << ' ';
}

float uif(uint32_t v) {
  float f;
  memcpy(&f, &v, 4);
  return f;
}

uint32_t fui(float f) {
  uint32_t ret;
  memcpy(&ret, &f, 4);
  return ret;
}

static void print_value(std::ostream &os, uint32_t value, int bits) {
  /* Guess if it's int or float */
  if (value <= (1 << 15)) {
    if (value <= 9)
      os << value << "\n";
    else
      os << value << " (0x" << std::hex << std::setw(bits / 4)
         << std::setfill('0') << value << std::dec << ")\n";
  } else {
    float f = uif(value);

    if (fabs(f) < 100000 && f * 10 == floor(f * 10)) {
      os.precision(1);
      os.setf(std::ios::fixed);
      os << f << " (0x" << std::hex << std::setw(bits / 4) << std::setfill('0')
         << value << std::dec << ")\n";
    } else
      /* Don't print more leading zeros than there are bits. */
      os << "0x" << std::hex << std::setw(bits / 4) << std::setfill('0')
         << value << std::dec << "\n";
  }
}

static const struct si_reg *find_register(const struct si_reg *table,
                                          unsigned table_size,
                                          unsigned offset)
{
        for (unsigned i = 0; i < table_size; i++) {
                const struct si_reg *reg = &table[i];

                if (reg->offset == offset)
                        return reg;
        }

        return NULL;
}

static const struct si_reg *find_gfx_reg(enum radeon_class chip_class, unsigned offset)
{
  const struct si_reg *reg = NULL;

  if (chip_class == GFX11)
    reg = find_register(gfx11_reg_table, ARRAY_SIZE(gfx11_reg_table), offset);
  else if (chip_class == GFX10)
    reg = find_register(gfx10_reg_table, ARRAY_SIZE(gfx10_reg_table), offset);
  else if (chip_class == GFX9)
    reg = find_register(gfx9_reg_table, ARRAY_SIZE(gfx9_reg_table), offset);
  else if (chip_class == GFX8)
    reg = find_register(gfx8_reg_table, ARRAY_SIZE(gfx8_reg_table), offset);
  else if (chip_class == GFX7)
    reg = find_register(gfx7_reg_table, ARRAY_SIZE(gfx7_reg_table), offset);
  else if (chip_class == GFX6)
    reg = find_register(gfx6_reg_table, ARRAY_SIZE(gfx6_reg_table), offset);
  return reg;
}
  

static void si_dump_reg(std::ostream &os, unsigned offset, uint32_t value,
                        uint32_t field_mask) {
  int r, f;

  const struct si_reg *reg = find_gfx_reg(chip_class, offset);
  
  if (reg) {
      bool first_field = true;
      const char *reg_name = sid_strings + reg->name_offset;

      print_spaces(os, INDENT_PKT);
      os << COLOR_YELLOW << reg_name << "," << std::hex << value << std::dec << " " << COLOR_RESET << " <- ";

      if (!reg->num_fields) {
        print_value(os, value, 32);
        return;
      }

      for (f = 0; f < reg->num_fields; f++) {
        const struct si_field *field = sid_fields_table + reg->fields_offset + f;
        const int *values_offsets = sid_strings_offsets + field->values_offset;
        uint32_t val = (value & field->mask) >> (ffs(field->mask) - 1);

        if (!(field->mask & field_mask))
          continue;

        /* Indent the field. */
        if (!first_field)
          print_spaces(os, INDENT_PKT + strlen(reg_name) + 4);

        /* Print the field. */
        os << sid_strings + field->name_offset << " = ";

        if (val < field->num_values && values_offsets[val] >= 0)
          os << sid_strings + values_offsets[val] << "\n";
        else
          print_value(os, val, __builtin_popcountll(field->mask));

        first_field = false;
      }
      return;
  }
  os << std::hex << "unknown reg " << offset << "  = " << value << std::dec << "\n";
}

static void print_named_value(std::ostream &os, const char *name,
                              uint32_t value, int bits) {
  print_spaces(os, INDENT_PKT);
  os << COLOR_YELLOW << name << COLOR_RESET " <- ";
  print_value(os, value, bits);
}

static void print_reg_name(std::ostream &os, int offset)
{
    int r, f;
    const struct si_reg *reg = find_gfx_reg(chip_class, offset);

  if (reg) {
    const char *reg_name = sid_strings + reg->name_offset;
    
    if (reg->offset == offset) {
      print_spaces(os, INDENT_PKT);
      os << COLOR_YELLOW << reg_name << COLOR_RESET << "\n";
    }
  }
}

static std::map<std::vector<std::uint32_t>, std::string> ls_shaders, hs_shaders,
    vs_shaders, ps_shaders, gs_shaders, es_shaders, cs_shaders;

std::string
dump_shader(std::map<std::vector<std::uint32_t>, std::string> &cache,
            std::string const &cat, std::uint64_t addr) {
  if (addr == 0xffff800000000000ull)
    return "unknown shader";
  uint32_t *data = (uint32_t *)get_ptr(addr, 0);
  if (!data) {
    fprintf(stderr, "Failed to dump shader\n");
    return "unknown shader";
  }

  uint32_t *end = data;
  while (*end != 0xBF810000) {
    ++end;
  }
  ++end;
  std::vector<uint32_t> body(data, end);
  auto it = cache.find(body);
  if (it != cache.end())
    return it->second;

  auto id = cache.size();
  std::string name = cat + "_shader_" + std::to_string(id);
  std::ofstream dump("/tmp/shader_binary");
  dump.write((char *)data, (end - data) * 4);
  dump.close();
  std::string cmd_line =
      "clrxdisasm -r -g Tonga /tmp/shader_binary > " + get_output_dir() + name + ".s";
  if (system(cmd_line.c_str())) {
    std::cerr << "failed to execute clrxdisasm" << std::endl;
  }
  cache[body] = name;
  return name;
}

static std::uint32_t config_reg;
void process_set_reg_mask(std::ostream &os, std::uint32_t reg, std::uint32_t value, std::uint32_t mask,
                          std::map<std::uint32_t, std::uint32_t>& registers) {
  reg &= 0xFFFFFFU;

  if (get_dump_draws()) {
    registers[reg] = (value & mask) | (registers[reg] & ~mask);
  }

  si_dump_reg(os, reg, value, mask);
}

std::int64_t get_shader_addr(std::uint32_t lo_value, std::uint32_t value)
{
  std::int64_t addr = (std::int32_t)value << 24;
  addr <<= 16;
  addr |= (static_cast<uint64_t>(lo_value) << 8);
  return addr;
}

void dump_shader_from_addr(std::ostream &os, std::map<std::vector<std::uint32_t>, std::string> &cache,
                           const char *nm, std::uint32_t lo_value, std::uint32_t value)
{
  std::int64_t s_addr = get_shader_addr(lo_value, value);
  if (s_addr) {
    auto s = dump_shader(cache, nm, s_addr);
    print_spaces(os, 8);
    os << s << "\n";
  }
}

void process_set_reg(std::ostream &os, std::uint32_t reg, std::uint32_t value,
                     std::map<std::uint32_t, std::uint32_t>& registers) {
  reg &= 0xFFFFFFU;
  static std::uint32_t lo_value;

  if (get_dump_draws()) {
    registers[reg] = value;
  }

  si_dump_reg(os, reg, value, 0xFFFFFFFFU);
  if (reg == R_00B420_SPI_SHADER_PGM_LO_HS)
    lo_value = value;
  if (reg == R_00B424_SPI_SHADER_PGM_HI_HS) {
    dump_shader_from_addr(os, hs_shaders, "hs", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B020_SPI_SHADER_PGM_LO_PS)
    lo_value = value;
  if (reg == R_00B024_SPI_SHADER_PGM_HI_PS) {
    dump_shader_from_addr(os, ps_shaders, "ps", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B120_SPI_SHADER_PGM_LO_VS)
    lo_value = value;
  if (reg == R_00B124_SPI_SHADER_PGM_HI_VS) {
    dump_shader_from_addr(os, vs_shaders, "vs", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B320_SPI_SHADER_PGM_LO_ES)
    lo_value = value;
  if (reg == R_00B324_SPI_SHADER_PGM_HI_ES) {
    dump_shader_from_addr(os, es_shaders, "es", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B220_SPI_SHADER_PGM_LO_GS)
    lo_value = value;
  if (reg == R_00B224_SPI_SHADER_PGM_HI_GS) {
    dump_shader_from_addr(os, gs_shaders, "gs", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B520_SPI_SHADER_PGM_LO_LS)
    lo_value = value;
  if (reg == R_00B524_SPI_SHADER_PGM_HI_LS) {
    dump_shader_from_addr(os, ls_shaders, "ls", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B830_COMPUTE_PGM_LO)
    lo_value = value;
  if (reg == R_00B834_COMPUTE_PGM_HI) {
    dump_shader_from_addr(os, cs_shaders, "cs", lo_value, value);
    lo_value = 0;
  }
  if (reg == R_00B450_SPI_SHADER_USER_DATA_HS_8)
    config_reg = value;
}

void dump_draw(const std::string& draw_description,
               const std::map<std::uint32_t, std::uint32_t>& registers)
{
  if (!get_dump_draws())
    return;

  static int draw_id = 0;

  std::ofstream out(get_output_dir() + "draw." + std::to_string(draw_id) + ".txt");

  out << "draw \"" << draw_description << "\"\n";

  for (auto e : registers)
    si_dump_reg(out, e.first, e.second, 0xFFFFFFFFU);
  ++draw_id;
}

void process_packet0(std::ostream &os, uint32_t const *packet,
                     std::map<std::uint32_t, std::uint32_t>& registers) {
  unsigned reg = PKT0_BASE_INDEX_G(*packet) * 4;
  unsigned cnt = PKT_COUNT_G(*packet) + 1;
  for (unsigned i = 0; i < cnt; ++i) {
    process_set_reg(os, reg + 4 * i, packet[1 + i], registers);
  }
}

static void process_ib(std::ostream &os, uint32_t *curr, uint32_t const *e, 
                       std::map<std::uint32_t, std::uint32_t>& registers);
static void process_dma_ib(std::ostream &os, uint32_t *curr, uint32_t const *e);
static void process_si_dma_ib(std::ostream &os, uint32_t *curr, uint32_t const *e);
static size_t cs_id = 0;

void process_packet3(std::ostream &os, uint32_t *packet, std::map<std::uint32_t, std::uint32_t>& registers) {
  auto op = PKT3_IT_OPCODE_G(*packet);
  auto pred = PKT3_PREDICATE(*packet);
  int i;

  /* Print the name first. */
  for (i = 0; i < ARRAY_SIZE(packet3_table); i++)
    if (packet3_table[i].op == op)
      break;
  if (i < ARRAY_SIZE(packet3_table)) {
    const char *name = sid_strings + packet3_table[i].name_offset;
    if (op == PKT3_SET_CONTEXT_REG || op == PKT3_SET_CONFIG_REG ||
        op == PKT3_SET_UCONFIG_REG || op == PKT3_SET_SH_REG || op == PKT3_SET_SH_REG_INDEX || op == PKT3_SET_UCONFIG_REG_INDEX) {
      auto idx = (packet[1] >> 28) & 0x7;
      char idx_str[5] = {0};
      if (idx)
        snprintf(idx_str, 5, "(%d)", idx);
      os << COLOR_CYAN << name << idx_str << COLOR_CYAN << (pred ? "(P)" :  "") << ":\n";
    } else
      os << COLOR_GREEN << name << COLOR_CYAN << (pred ? "(P)" :  "") << ":\n";
  }
  /*else
          fprintf(f, COLOR_RED "PKT3_UNKNOWN 0x%x%s" COLOR_RESET ":\n",
                  op, predicate);*/

  switch (PKT3_IT_OPCODE_G(*packet)) {
  case PKT3_SET_CONTEXT_REG: {
    unsigned reg = packet[1] * 4 + SI_CONTEXT_REG_OFFSET;
    for (unsigned i = 0; i < PKT_COUNT_G(packet[0]); ++i) {
      process_set_reg(os, reg + 4 * i, packet[2 + i], registers);
    }
  } break;
  case PKT3_LOAD_CONTEXT_REG: {
    unsigned reg = (packet[3] & 0xffff) * 4 + SI_CONTEXT_REG_OFFSET;
    print_named_value(os, "ADDR_LO", packet[1], 32);
    print_named_value(os, "ADDR_HI", packet[2] & 0xffff, 32);
    print_named_value(os, "NUM_DWORDS", packet[4] & 0xffff, 32);
    print_reg_name(os, packet[3] * 4 + SI_CONTEXT_REG_OFFSET);
  } break;
  case PKT3_LOAD_CONTEXT_REG_INDEX: {
    unsigned base_reg = (packet[3] & 0xffff) * 4 + SI_CONTEXT_REG_OFFSET;
    print_named_value(os, "ADDR_LO", packet[1] & ~0x3, 32);
    print_named_value(os, "ADDR_HI", packet[2] & 0xffff, 32);
    print_named_value(os, "NUM_DWORDS", packet[4] & 0xffff, 32);
    bool load_index = !!(packet[3] & (1 << 31));
    std::uint64_t va = static_cast<std::uint64_t>(packet[2]) << 32;
    va |= (packet[1] & ~0x3);
    std::uint32_t num_dwords = (packet[4] & 0xffff) * (load_index ? 2 : 1);
    uint32_t *data = (uint32_t *)get_ptr(va, num_dwords << 2);
    if (data) {
      if (load_index) {
        for (unsigned i = 0; i < num_dwords; i+=2) {
          unsigned reg = data[i] * 4 + SI_CONTEXT_REG_OFFSET;
          process_set_reg(os, reg, data[i + 1], registers);
        }
      } else {
        for (unsigned i = 0; i < num_dwords; i++) {
          unsigned reg = base_reg + i * 4;
          process_set_reg(os, reg, data[i], registers);
        }
      }
    } else if (!load_index)
      print_reg_name(os, base_reg);
  } break;
  case PKT3_LOAD_SH_REG_INDEX: {
    unsigned base_reg = (packet[3] & 0xffff) * 4 + SI_SH_REG_OFFSET;
    print_named_value(os, "ADDR_LO", packet[1] & ~0x3, 32);
    print_named_value(os, "ADDR_HI", packet[2] & 0xffff, 32);
    print_named_value(os, "NUM_DWORDS", packet[4] & 0x3fff, 32);
    bool load_index = !!(packet[3] & (1 << 31));
    std::uint64_t va = static_cast<std::uint64_t>(packet[2]) << 32;
    va |= (packet[1] & ~0x3);
    std::uint32_t num_dwords = (packet[4] & 0xffff) * (load_index ? 2 : 1);
    uint32_t *data = (uint32_t *)get_ptr(va, num_dwords << 2);
    if (data) {
      if (load_index) {
        for (unsigned i = 0; i < num_dwords; i+=2) {
          unsigned reg = data[i] * 4 + SI_SH_REG_OFFSET;
          process_set_reg(os, reg, data[i + 1], registers);
        }
      } else {
        for (unsigned i = 0; i < num_dwords; i++) {
          unsigned reg = base_reg + i * 4;
          process_set_reg(os, reg, data[i], registers);
        }
      }
    } else if (!load_index)
      print_reg_name(os, base_reg);
  } break;
  case PKT3_SET_SH_REG: {
    unsigned reg = packet[1] * 4 + SI_SH_REG_OFFSET;
    for (unsigned i = 0; i < PKT_COUNT_G(packet[0]); ++i) {
      process_set_reg(os, reg + 4 * i, packet[2 + i], registers);
    }
  } break;
  case PKT3_SET_SH_REG_INDEX: {
    unsigned reg = packet[1] * 4 + SI_SH_REG_OFFSET;
    for (unsigned i = 0; i < PKT_COUNT_G(packet[0]); ++i) {
      process_set_reg(os, reg + 4 * i, packet[2 + i], registers);
    }
  } break;
  case PKT3_SET_CONFIG_REG: {
    unsigned reg = packet[1] * 4 + SI_CONFIG_REG_OFFSET;
    for (unsigned i = 0; i < PKT_COUNT_G(packet[0]); ++i) {
      process_set_reg(os, reg + 4 * i, packet[2 + i], registers);
    }
  } break;
  case PKT3_SET_UCONFIG_REG:
  case PKT3_SET_UCONFIG_REG_INDEX: {
    unsigned reg = packet[1] * 4 + CIK_UCONFIG_REG_OFFSET;
    for (unsigned i = 0; i < PKT_COUNT_G(packet[0]); ++i) {
      process_set_reg(os, reg + 4 * i, packet[2 + i], registers);
    }
  } break;
  case PKT3_CONTEXT_CONTROL:
    print_named_value(os, "LOAD_CONTROL", packet[1], 32);
    print_named_value(os, "SHADOW_CONTROL", packet[2], 32);
    break;
  case PKT3_ACQUIRE_MEM:
    si_dump_reg(os, R_0301F0_CP_COHER_CNTL, packet[1], ~0);
    si_dump_reg(os, R_0301F4_CP_COHER_SIZE, packet[2], ~0);
    si_dump_reg(os, R_030230_CP_COHER_SIZE_HI, packet[3], ~0);
    si_dump_reg(os, R_0301F8_CP_COHER_BASE, packet[4], ~0);
    si_dump_reg(os, R_0301E4_CP_COHER_BASE_HI, packet[5], ~0);
    print_named_value(os, "POLL_INTERVAL", packet[6], 16);
    break;
  case PKT3_SURFACE_SYNC:
    si_dump_reg(os, R_0085F0_CP_COHER_CNTL, packet[1], ~0);
    si_dump_reg(os, R_0085F4_CP_COHER_SIZE, packet[2], ~0);
    si_dump_reg(os, R_0085F8_CP_COHER_BASE, packet[3], ~0);
    print_named_value(os, "POLL_INTERVAL", packet[4], 16);
    break;
  case PKT3_EVENT_WRITE:
    si_dump_reg(os, R_028A90_VGT_EVENT_INITIATOR, packet[1],
                S_028A90_EVENT_TYPE(~0));
    print_named_value(os, "EVENT_INDEX", (packet[1] >> 8) & 0xf, 4);
    print_named_value(os, "INV_L2", (packet[1] >> 20) & 0x1, 1);
    if (PKT_COUNT_G(packet[0]) > 0) {
      print_named_value(os, "ADDRESS_LO", packet[2], 32);
      print_named_value(os, "ADDRESS_HI", packet[3], 16);
    }
    break;
  case PKT3_EVENT_WRITE_EOP:
    si_dump_reg(os, R_028A90_VGT_EVENT_INITIATOR, packet[1],
                S_028A90_EVENT_TYPE(~0));
    print_named_value(os, "EVENT_INDEX", (packet[1] >> 8) & 0xf, 3);
    print_named_value(os, "DATA_SEL", (packet[3] >> 29) & 0x7, 3);
    print_named_value(os, "ADDR_LO", packet[2], 32);
    print_named_value(os, "ADDR_HI", packet[3] & 0xffff, 32);
    print_named_value(os, "SEQ_LO", packet[4], 32);
    print_named_value(os, "SEQ_HI", packet[5], 32);
    break;
  case PKT3_RELEASE_MEM:
    si_dump_reg(os, R_028A90_VGT_EVENT_INITIATOR, packet[1],
                S_028A90_EVENT_TYPE(~0));
    print_named_value(os, "EVENT_INDEX", (packet[1] >> 8) & 0xf, 4);
    print_named_value(os, "DATA_SEL", (packet[2] >> 29) & 0x7, 3);
    print_named_value(os, "ADDR_LO", packet[3], 32);
    print_named_value(os, "ADDR_HI", packet[4], 32);
    print_named_value(os, "SEQ_LO", packet[5], 32);
    print_named_value(os, "SEQ_HI", packet[6], 32);
    break;
  case PKT3_DRAW_INDEX_AUTO:
    process_set_reg_mask(os, R_030930_VGT_NUM_INDICES, packet[1], ~0, registers);
    process_set_reg_mask(os, R_0287F0_VGT_DRAW_INITIATOR, packet[2], ~0, registers);
    dump_draw("PKT3_DRAW_INDEX_AUTO", registers);
    break;
  case PKT3_DRAW_INDEX_2:
    process_set_reg_mask(os, R_028A78_VGT_DMA_MAX_SIZE, packet[1], ~0, registers);
    process_set_reg_mask(os, R_0287E8_VGT_DMA_BASE, packet[2], ~0, registers);
    process_set_reg_mask(os, R_0287E4_VGT_DMA_BASE_HI, packet[3], ~0, registers);
    process_set_reg_mask(os, R_030930_VGT_NUM_INDICES, packet[4], ~0, registers);
    process_set_reg_mask(os, R_0287F0_VGT_DRAW_INITIATOR, packet[5], ~0, registers);
    dump_draw("PKT3_DRAW_INDEX_2", registers);
    break;
  case PKT3_INDEX_TYPE:
    si_dump_reg(os, R_028A7C_VGT_DMA_INDEX_TYPE, packet[1], ~0);
    if (get_dump_draws()) {
      registers[R_028A7C_VGT_DMA_INDEX_TYPE] = packet[1];
    }
    break;
  case PKT3_NUM_INSTANCES:
    si_dump_reg(os, R_030934_VGT_NUM_INSTANCES, packet[1], ~0);
    if (get_dump_draws()) {
      registers[R_030934_VGT_NUM_INSTANCES] = packet[1];
    }
    break;
  case PKT3_WRITE_DATA:
    si_dump_reg(os, R_370_CONTROL, packet[1], ~0);
    si_dump_reg(os, R_371_DST_ADDR_LO, packet[2], ~0);
    si_dump_reg(os, R_372_DST_ADDR_HI, packet[3], ~0);
    if (packet[2] == config_reg) {
      packet[11] = fui(128.0);
      std::cerr << "write config_reg " << std::hex << packet[11] << " "
                << std::dec << uif(packet[11]) << "\n";
    }
    for (unsigned i = 2; i < PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[2 + i] << std::dec << "\n";
    }
    break;
  case PKT3_CP_DMA:
    si_dump_reg(os, R_410_CP_DMA_WORD0, packet[1], ~0);
    si_dump_reg(os, R_411_CP_DMA_WORD1, packet[2], ~0);
    si_dump_reg(os, R_412_CP_DMA_WORD2, packet[3], ~0);
    si_dump_reg(os, R_413_CP_DMA_WORD3, packet[4], ~0);
    si_dump_reg(os, R_415_COMMAND, packet[5], ~0);
    break;
  case PKT3_DMA_DATA:
    si_dump_reg(os, R_500_DMA_DATA_WORD0, packet[1], ~0);
    si_dump_reg(os, R_501_SRC_ADDR_LO, packet[2], ~0);
    si_dump_reg(os, R_502_SRC_ADDR_HI, packet[3], ~0);
    si_dump_reg(os, R_503_DST_ADDR_LO, packet[4], ~0);
    si_dump_reg(os, R_504_DST_ADDR_HI, packet[5], ~0);
    si_dump_reg(os, R_415_COMMAND, packet[6], ~0);
    break;
  case PKT3_COPY_DATA:
    print_named_value(os, "SRC_SEL", (packet[1] >> 0) & 0xf, 4);
    print_named_value(os, "DST_SEL", (packet[1] >> 8) & 0xf, 4);
    print_named_value(os, "COUNT_SEL", (packet[1] >> 16) & 1, 1);
    print_named_value(os, "WR_CONFIRM", (packet[1] >> 20) & 1, 1);
    print_named_value(os, "ENGINE_SEL", (packet[1] >> 30) & 3, 2);
    print_named_value(os, "SRC_ADDR_LO", packet[2], 32);
    print_named_value(os, "SRC_ADDR_HI", packet[3], 32);
    print_named_value(os, "DST_ADDR_LO", packet[4], 32);
    print_named_value(os, "DST_ADDR_HI", packet[5], 32);
    break;
  case PKT3_INCREMENT_CE_COUNTER:
    print_named_value(os, "CE_COUNTER_DUMMY", packet[1], 32);
    for (unsigned i = 1; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
    }
    break;
  case PKT3_INCREMENT_DE_COUNTER:
    print_named_value(os, "DE_COUNTER_DUMMY", packet[1], 32);
    for (unsigned i = 1; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
    }
    break;
  case PKT3_WAIT_ON_CE_COUNTER:
    print_named_value(os, "WAIT_CE_COUNTER_DUMMY", packet[1], 32);
    for (unsigned i = 1; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
    }
    break;
  case PKT3_DUMP_CONST_RAM:
    print_named_value(os, "OFFSET", packet[1], 32);
    print_named_value(os, "SIZE", packet[2], 32);
    print_named_value(os, "ADDR_LO", packet[3], 32);
    print_named_value(os, "ADDR_HI", packet[4], 32);
    for (unsigned i = 4; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "warn 0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[1 + i] << std::dec << "\n";
    }
    break;
  case PKT3_LOAD_CONST_RAM:
    print_named_value(os, "ADDR_LO", packet[1], 32);
    print_named_value(os, "ADDR_HI", packet[2], 32);
    print_named_value(os, "SIZE", packet[3], 32);
    print_named_value(os, "OFFSET", packet[4], 32);
    for (unsigned i = 4; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "warn 0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[1 + i] << std::dec << "\n";
    }
    break;
  case PKT3_WRITE_CONST_RAM:
    print_named_value(os, "OFFSET", packet[1], 32);
    for (unsigned i = 0; i < PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[2 + i] << std::dec << "\n";
    }
    break;
  case PKT3_DRAW_INDEX_INDIRECT_MULTI:
    for (unsigned i = 0; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[1 + i] << std::dec << "\n";
    }
    dump_draw("PKT3_DRAW_INDEX_INDIRECT_MULTI", registers);
    break;
  case PKT3_SET_BASE:
    for (unsigned i = 0; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[1 + i] << std::dec << "\n";
    }
    break;
  default:
  case PKT3_INDEX_BASE:
    for (unsigned i = 0; i <= PKT_COUNT_G(packet[0]); i++) {
      print_spaces(os, INDENT_PKT);
      os << "0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[1 + i] << std::dec << "\n";
    }
    break;
  case PKT3_INDIRECT_BUFFER_CIK:
  case PKT3_INDIRECT_BUFFER_CONST: {
    print_named_value(os, "IB_BASE_LO", packet[1], 32);
    print_named_value(os, "IB_BASE_HI", packet[2], 32);
    print_named_value(os, "IB_SIZE", packet[3] & 0xFFFFF, 20);
    print_named_value(os, "CHAIN", (packet[3] >> 20) & 1, 1);
    print_named_value(os, "VALID", (packet[3] >> 23) & 1, 1);
    std::uint64_t va = static_cast<std::uint64_t>(packet[2]) << 32;
    va |= packet[1];
    unsigned words = packet[3] & 0xfffff;
    uint32_t *data = (uint32_t *)get_ptr(va, words * 4);
    process_ib(os, data, data + words, registers);
  } break;
  case PKT3_NOP:
    os << "     trace id: 0x" << std::setw(8) << std::setfill('0') << std::hex
         << packet[1] << std::dec << "\n";
  }
}

static void process_ib(std::ostream &os, uint32_t *curr, uint32_t const *e,
                       std::map<std::uint32_t, std::uint32_t>& registers) {
  while (curr != e) {
    if (curr > e) {
      std::cerr << "went past end of IB at CS " << cs_id << ": " << std::hex << curr << " " << e
                << std::endl;
      abort();
    }
    switch (PKT_TYPE_G(*curr)) {
    case 0:
      process_packet0(os, curr, registers);
      curr += 2 + PKT_COUNT_G(*curr);
      break;
    case 2:
      curr += 1;
      break;
    case 3:
      if (*curr == 0xffff1000u) {
        ++curr;
        break;
      }
      process_packet3(os, curr, registers);
      curr += 2 + PKT_COUNT_G(*curr);
      break;
    default:
      os << "unknown packet type " << PKT_TYPE_G(*curr) << std::hex << " "
         << *curr << std::dec << "\n";
      abort();
    }
  }
}

static void process_si_dma_ib(std::ostream &os, uint32_t *curr, uint32_t const *e) {
  while (curr != e) {
    if (curr > e) {
      std::cerr << "went past end of IB at CS " << cs_id << ": " << std::hex << curr << " " << e
                << std::endl;
      abort();
    }
    uint32_t val = curr[0];
    uint32_t op = (val >> 28) & 0xf;
    uint32_t size = (val & 0xffffff);
    uint32_t pkt_count;
    switch (op) {
    case SI_DMA_PACKET_NOP:
      ++curr;
      os << "DMA NOP" << "\n";
      break;
    case SI_DMA_PACKET_CONSTANT_FILL:
      os << "DMA CONSTANT FILL: " << size << "\n";
      print_named_value(os, "ADDR_LO", curr[1], 32);
      print_named_value(os, "DATA", curr[2], 32);
      print_named_value(os, "ADDR_HI", curr[3], 32);
      curr += 4;
      break;
    case SI_DMA_PACKET_WRITE:
      os << "DMA WRITE" << "\n";
      print_named_value(os, "NUM_DWORDS", size, 32);
      print_named_value(os, "DST_ADDR_LO", curr[1], 32);
      print_named_value(os, "DST_ADDR_HI", curr[2], 32);
      pkt_count = size + 3;
      for (unsigned i = 0; i < size; i++) {
          print_spaces(os, INDENT_PKT);
          os << "0x" << std::setw(8) << std::setfill('0') << std::hex
             << curr[4 + i] << std::dec << "\n";
      }
      curr += pkt_count;
      break;
    case SI_DMA_PACKET_COPY: {
      uint32_t sub_op = (val >> 20) & 0xff;
      switch (sub_op) {
      case 0x00:
      case 0x40:
        os << "DMA COPY" << ((sub_op == 0x40) ? " BYTE" : "") << "\n";
        print_named_value(os, "NUM_DWORDS", size, 32);
        print_named_value(os, "DST_ADDR_LO", curr[1], 32);
        print_named_value(os, "SRC_ADDR_LO", curr[2], 32);
        print_named_value(os, "DST_ADDR_HI", curr[3], 32);
        print_named_value(os, "SRC_ADDR_HI", curr[4], 32);
        curr += 5;
        break;
      case 0x41:
        os << "DMA COPY LINEAR 0x" << std::hex << curr[0] << std::dec << "\n";
        print_named_value(os, "SRC_ADDR_LO", curr[1], 32);
        print_named_value(os, "SRC_ADDR_HI_PITCH", curr[2], 32);
        print_named_value(os, "SRC_SLICE_PITCH", curr[3], 32);
        print_named_value(os, "DST_ADDR_LO", curr[4], 32);
        print_named_value(os, "DST_ADDR_HI_PITCH", curr[5], 32);
        print_named_value(os, "DST_SLICE_PITCH", curr[6], 32);
        print_named_value(os, "XY", curr[7], 32);
        print_named_value(os, "Z", curr[8], 32);
        curr += 9;
        break;
      case 0x49:
        os << "DMA COPY L2T 0x" << std::hex << curr[0] << std::dec << "\n";
        print_named_value(os, "TILE_ADDR_LO", curr[1], 32);
        print_named_value(os, "TILE_INFO0", curr[2], 32);
        print_named_value(os, "TILE_PITCH_TILE_MAX", curr[3], 32);
        print_named_value(os, "TILE_INFO2", curr[4], 32);
        print_named_value(os, "TILE_INFO3", curr[5], 32);
        print_named_value(os, "TILE_INFO4", curr[6], 32);
        print_named_value(os, "LIN_ADDR_LO", curr[7], 32);
        print_named_value(os, "LIN_ADDR_HI_PITCH", curr[8], 32);
        print_named_value(os, "LIN_SLICE_PITCH", curr[9], 32);
        print_named_value(os, "XY", curr[10], 32);
        print_named_value(os, "Z", curr[11], 32);
        curr += 12;
        break;
      case 0x4d:
        os << "DMA COPY T2T 0x" << std::hex << curr[0] << std::dec << "\n";
        print_named_value(os, "SRC_ADDR", curr[1], 32);
        print_named_value(os, "SRC_INFO1", curr[2], 32);
        print_named_value(os, "SRC_INFO2", curr[3], 32);
        print_named_value(os, "DST_ADDR", curr[4], 32);
        print_named_value(os, "DST_INFO1", curr[5], 32);
        print_named_value(os, "DST_INFO2", curr[6], 32);
        print_named_value(os, "INFO0", curr[7], 32);
        print_named_value(os, "xINFO0", curr[8], 32);
        print_named_value(os, "yINFO1", curr[9], 32);
        print_named_value(os, "zINFO2", curr[10], 32);
        print_named_value(os, "dINFO1", curr[11], 32);
        print_named_value(os, "dzINFO13", curr[12], 32);
        curr += 13;
        break;
      }
      break;
    }
    default:
      os << "DMA UNKNOWN 0x" << std::hex << curr[0] << std::dec << "\n";
      curr++;
      break;
    }
  }
}

static void process_dma_ib(std::ostream &os, uint32_t *curr, uint32_t const *e) {
  while (curr != e) {
    if (curr > e) {
      std::cerr << "went past end of IB at CS " << cs_id << ": " << std::hex << curr << " " << e
                << std::endl;
      abort();
    }
    uint32_t val = curr[0];
    uint32_t op = val & 0xff;
    uint32_t pkt_count;
    switch (op) {
    case CIK_SDMA_OPCODE_NOP:
      ++curr;
      os << "DMA NOP" << "\n";
      break;
    case CIK_SDMA_OPCODE_COPY: {
      uint32_t sub_op = (val >> 8) & 0xff;
      switch (sub_op) {
      case CIK_SDMA_COPY_SUB_OPCODE_LINEAR:
        pkt_count = 7;
        os << "DMA COPY LINEAR" << "\n";
        print_named_value(os, "SIZE", curr[1], 32);
        print_named_value(os, "OFFSET", curr[2], 32);
        print_named_value(os, "SRC_ADDR_LO", curr[3], 32);
        print_named_value(os, "SRC_ADDR_HI", curr[4], 32);
        print_named_value(os, "DST_ADDR_LO", curr[5], 32);
        print_named_value(os, "DST_ADDR_HI", curr[6], 32);
        break;
      case CIK_SDMA_COPY_SUB_OPCODE_TILED:
        pkt_count = 12;
        os << "DMA COPY TILED" << "\n";
        print_named_value(os, "TILED_ADDR_LO", curr[1], 32);
        print_named_value(os, "TILED_ADDR_HI", curr[2], 32);
        print_named_value(os, "DW_3", curr[3], 32);
        print_named_value(os, "SLICE_PITCH", curr[4], 32);
        print_named_value(os, "DW_5", curr[5], 32);
        print_named_value(os, "DW_6", curr[6], 32);
        print_named_value(os, "DW_7", curr[7], 32);
        print_named_value(os, "LINEAR_ADDR_LO", curr[8], 32);
        print_named_value(os, "LINEAR_ADDR_HI", curr[9], 32);
        print_named_value(os, "LINEAR_PITCH", curr[10], 32);
        print_named_value(os, "COUNT", curr[11], 32);        
        break;
      case CIK_SDMA_COPY_SUB_OPCODE_LINEAR_SUB_WINDOW:
        pkt_count = 13;
        os << "DMA COPY LINEAR SUB WINDOW 0x" << std::hex << curr[0] << std::dec << "\n";
        print_named_value(os, "SRC_ADDR_LO", curr[1], 32);
        print_named_value(os, "SRC_ADDR_HI", curr[2], 32);
        print_named_value(os, "SRC_XY", curr[3], 32);
        print_named_value(os, "SRC_PITCH", curr[4], 32);
        print_named_value(os, "SRC_SLICE_PITCH", curr[5], 32);
        print_named_value(os, "DST_ADDR_LO", curr[6], 32);
        print_named_value(os, "DST_ADDR_HI", curr[7], 32);
        print_named_value(os, "DST_XY", curr[8], 32);
        print_named_value(os, "DST_Z_PITCH", curr[9], 32);
        print_named_value(os, "DST_SLICE_PITCH", curr[10], 32);
        print_named_value(os, "W_H", curr[11], 32);
        print_named_value(os, "DEPTH", curr[12], 32);
        break;
      case CIK_SDMA_COPY_SUB_OPCODE_TILED_SUB_WINDOW:
        pkt_count = 14;
        os << "DMA COPY TILED SUB WINDOW 0x" << std::hex << curr[0] << std::dec << "\n";
        print_named_value(os, "X_ADDR_LO", curr[1], 32);
        print_named_value(os, "X_ADDR_HI", curr[2], 32);
        print_named_value(os, "X_XY", curr[3], 32);
        print_named_value(os, "X_PITCH", curr[4], 32);
        print_named_value(os, "X_SRC_SLICE_PITCH", curr[5], 32);
        print_named_value(os, "TILE_INFO", curr[6], 32);
        print_named_value(os, "Y_ADDR_LO", curr[7], 32);
        print_named_value(os, "Y_ADDR_HI", curr[8], 32);
        print_named_value(os, "Y_XY", curr[9], 32);
        print_named_value(os, "Y_Z_PITCH", curr[10], 32);
        print_named_value(os, "Y_SLICE_PITCH", curr[11], 32);
        print_named_value(os, "W_H", curr[12], 32);
        print_named_value(os, "DEPTH", curr[13], 32);
        break;
      case CIK_SDMA_COPY_SUB_OPCODE_T2T_SUB_WINDOW:
        pkt_count = 15;
        os << "DMA COPY T2T SUB WINDOW 0x" << std::hex << curr[0] << std::dec << "\n";
        print_named_value(os, "SRC_ADDR_LO", curr[1], 32);
        print_named_value(os, "SRC_ADDR_HI", curr[2], 32);
        print_named_value(os, "SRC_XY", curr[3], 32);
        print_named_value(os, "SRC_PITCH", curr[4], 32);
        print_named_value(os, "SRC_SLICE_PITCH", curr[5], 32);
        print_named_value(os, "SRC_TILE_INFO", curr[6], 32);
        print_named_value(os, "DST_ADDR_LO", curr[7], 32);
        print_named_value(os, "DST_ADDR_HI", curr[8], 32);
        print_named_value(os, "DST_XY", curr[9], 32);
        print_named_value(os, "DST_Z_PITCH", curr[10], 32);
        print_named_value(os, "DST_SLICE_PITCH", curr[11], 32);
        print_named_value(os, "DST_TILE_INFO", curr[12], 32);
        print_named_value(os, "W_H", curr[13], 32);
        print_named_value(os, "DEPTH", curr[14], 32);
        break;
      default:
        os << "DMA COPY UNKNOWN" << "\n";
        break;
      }
      curr += pkt_count;
      break;
    }
    case CIK_SDMA_OPCODE_WRITE: {
      uint32_t sub_op = (val >> 8) & 0xff;
      switch (sub_op) {
      case SDMA_WRITE_SUB_OPCODE_LINEAR:
        os << "DMA WRITE LINEAR" << "\n";
        print_named_value(os, "DST_ADDR_LO", curr[1], 32);
        print_named_value(os, "DST_ADDR_HI", curr[2], 32);
        print_named_value(os, "NUM_DWORDS", curr[3], 32);
        pkt_count = curr[3] + 4;
        for (unsigned i = 0; i < curr[3]; i++) {
          print_spaces(os, INDENT_PKT);
          os << "0x" << std::setw(8) << std::setfill('0') << std::hex
             << curr[4 + i] << std::dec << "\n";
        }
        break;
      default:
        os << "DMA WRITE UNKNOWN" << "\n";
        break;
      }
      curr += pkt_count;
      break;
    }
    case CIK_SDMA_OPCODE_INDIRECT_BUFFER:
      curr += 6;
      break;
    case CIK_SDMA_PACKET_CONSTANT_FILL:
      os << "DMA CONSTANT FILL" << "\n";
      print_named_value(os, "ADDR_LO", curr[1], 32);
      print_named_value(os, "ADDR_HI", curr[2], 32);
      print_named_value(os, "DATA", curr[3], 32);
      print_named_value(os, "FILLSIZE", curr[4], 32);
      curr += 5;
      break;
    case CIK_SDMA_OPCODE_TIMESTAMP:
      os << "DMA TIMESTAMP" << "\n";
      curr += 3;
      break;
    default:
      os << "DMA UNKNOWN 0x" << std::hex << curr[0] << std::dec << "\n";
      curr++;
      break;
    }
  }
}

int amdgpu_cs_submit(amdgpu_context_handle context, uint64_t flags,
                     struct amdgpu_cs_request *ibs_request,
                     uint32_t number_of_requests) {

  std::map<std::uint32_t, std::uint32_t> registers;
  for (unsigned i = 0; i < number_of_requests; ++i) {
    for (unsigned j = 0; j < ibs_request[i].number_of_ibs; ++j) {
      auto addr = ibs_request[i].ibs[j].ib_mc_address;
      auto size = ibs_request[i].ibs[j].size;
      std::ofstream out0(get_output_dir() + "cs." + std::to_string(cs_id) +
                         ".type.txt");
      out0 << ibs_request[i].ip_type << "\n";
      uint32_t *data = (uint32_t *)get_ptr(addr, size * 4);
      if (data) {
        std::string cs_type = "unknown";
        if (ibs_request[i].ip_type == AMDGPU_HW_IP_DMA)
          cs_type = "dma";
        else if (ibs_request[i].ibs[j].flags == 0)
          cs_type = "de";
        else if (ibs_request[i].ibs[j].flags == 1)
          cs_type = "ce";
        else if (ibs_request[i].ibs[j].flags == 3)
          cs_type = "ce_preamble";

        std::ofstream out(get_output_dir() + "cs." + std::to_string(cs_id) + "." +
                          cs_type + ".txt");
        out << std::hex << addr << std::dec << "\n";

        if (ibs_request[i].ip_type == AMDGPU_HW_IP_DMA)
          process_si_dma_ib(out, data, data + size);
        else
          process_ib(out, data, data + size, registers);
      } else
        fprintf(stderr, "failed to get IB data\n");
    }
    ++cs_id;
  }

  return get_vtable().fn_cs_submit(context, flags, ibs_request, number_of_requests);
}

int amdgpu_cs_submit_raw(amdgpu_device_handle device,
                         amdgpu_context_handle context,
                         amdgpu_bo_list_handle resources,
                         int num_chunks,
                         struct drm_amdgpu_cs_chunk *chunks,
                         uint64_t *seq_no)
{
  fprintf(stderr, "submit bo list: %p num chunks: %d\n", resources, num_chunks);
  std::map<std::uint32_t, std::uint32_t> registers;
  for (unsigned i = 0; i < num_chunks; ++i) {
    struct drm_amdgpu_cs_chunk_data *chunk_data;
    switch(chunks[i].chunk_id) {
    case AMDGPU_CHUNK_ID_IB: {
      chunk_data = (struct drm_amdgpu_cs_chunk_data *)(uintptr_t)chunks[i].chunk_data;
      std::uint64_t addr = chunk_data->ib_data.va_start;
      std::uint64_t size = chunk_data->ib_data.ib_bytes / 4;
      fprintf(stderr, "    IB: %lx %lx type: %s:%d:%d flags: %x\n", addr, size, chunk_data->ib_data.ip_type == 0 ? "gfx" : chunk_data->ib_data.ip_type == 1 ? "compute" : chunk_data->ib_data.ip_type == 2 ? "sdma" : "unknown", chunk_data->ib_data.ip_instance, chunk_data->ib_data.ring, chunk_data->ib_data.flags);
      break;
    }
    default:
      fprintf(stderr, "    chunk %d\n", chunks[i].chunk_id);
      break;
    }
#if 0
    chunk_data = (struct drm_amdgpu_cs_chunk_data *)(uintptr_t)chunks[i].chunk_data;
    auto addr = chunk_data->ib_data.va_start;
    auto size = chunk_data->ib_data.ib_bytes / 4;

    std::ofstream out0(get_output_dir() + "cs." + std::to_string(cs_id) +
                       ".type.txt");
    out0 << chunk_data->ib_data.ip_type << "\n";
    uint32_t *data = (uint32_t *)get_ptr(addr, size * 4);
    if (data) {
      std::string cs_type = "unknown";
      if (chunk_data->ib_data.ip_type == AMDGPU_HW_IP_DMA)
        cs_type = "dma";
      else if (chunk_data->ib_data.flags == 0)
        cs_type = "de";
      else if (chunk_data->ib_data.flags == 1)
        cs_type = "ce";
      else if (chunk_data->ib_data.flags == 3)
        cs_type = "ce_preamble";

      std::ofstream out(get_output_dir() + "cs." + std::to_string(cs_id) + "." +
                        cs_type + ".txt");
      out << std::hex << addr << std::dec << "\n";
      if (chunk_data->ib_data.ip_type == AMDGPU_HW_IP_DMA)
        process_si_dma_ib(out, data, data + size);
      else
        process_ib(out, data, data + size, registers);
    }
    ++cs_id;
#endif
  }
  if (resources) {
    fprintf(stderr, "  bo list: %p\n", resources);
    std::unique_lock<std::mutex> lock(global_mutex);
    auto it = bo_lists.find(resources);
    if (it != bo_lists.end()) {
      for (auto e : it->second) {
        auto it2 = buffers.find(e.first);
        if(it2 == buffers.end()) {
          fprintf(stderr, "    bo: %p prio: %d\n", e.first, e.second);
          continue;
        }
        fprintf(stderr, "     bo: %p (size: %ld, align: %ld, heaps: ", e.first, it2->second.request.alloc_size, it2->second.request.phys_alignment);
        for (unsigned i = 1; i <= it2->second.request.preferred_heap; i <<= 1) {
          if (!(i & it2->second.request.preferred_heap))
            continue;
          if ((i - 1) & it2->second.request.preferred_heap)
            fprintf(stderr, "|");
          switch(i) {
            case 0x1:
              fprintf(stderr, "cpu");
              break;
            case 0x2:
              fprintf(stderr, "gtt");
              break;
            case 0x4:
              fprintf(stderr, "vram");
              break;
            case 0x8:
              fprintf(stderr, "gds");
              break;
            case 0x10:
              fprintf(stderr, "gws");
              break;
            case 0x20:
              fprintf(stderr, "oa");
              break;
          }
        }
        fprintf(stderr, ", flags: ");
        for (uint64_t i = 1; i <= it2->second.request.flags; i <<= 1) {
          if (!(i & it2->second.request.flags))
            continue;
          if ((i - 1) & it2->second.request.flags)
            fprintf(stderr, "|");
          switch(i) {
            case 0x1:
              fprintf(stderr, "cpu");
              break;
            case 0x2:
              fprintf(stderr, "nocpu");
              break;
            case 0x4:
              fprintf(stderr, "uswc");
              break;
            case 0x8:
              fprintf(stderr, "clear");
              break;
            case 0x10:
              fprintf(stderr, "shadow");
              break;
            case 0x20:
              fprintf(stderr, "contig");
              break;
            case 0x40:
              fprintf(stderr, "valid");
              break;
            case 0x80:
              fprintf(stderr, "explicit");
              break;
            case 0x100:
              fprintf(stderr, "mqd");
              break;
            case 0x200:
              fprintf(stderr, "wipe");
              break;
            case 0x1000:
              fprintf(stderr, "exported");
              break;
          }
        }
        fprintf(stderr, ") prio: %d\n", e.second);
      }
    }
    
  }
  return get_vtable().fn_cs_submit_raw(device, context, resources, num_chunks, chunks, seq_no);
}

int amdgpu_cs_submit_raw2(amdgpu_device_handle device,
                         amdgpu_context_handle context,
                         uint32_t bo_list_handle,
                         int num_chunks,
                         struct drm_amdgpu_cs_chunk *chunks,
                         uint64_t *seq_no)
{
  std::map<std::uint32_t, std::uint32_t> registers;
  for (unsigned i = 0; i < num_chunks; ++i) {
    struct drm_amdgpu_cs_chunk_data *chunk_data;
    if (chunks[i].chunk_id != AMDGPU_CHUNK_ID_IB)
      continue;
#if 1
    chunk_data = (struct drm_amdgpu_cs_chunk_data *)(uintptr_t)chunks[i].chunk_data;
    auto addr = chunk_data->ib_data.va_start;
    auto size = chunk_data->ib_data.ib_bytes / 4;

    std::ofstream out0(get_output_dir() + "cs." + std::to_string(cs_id) +
                       ".type.txt");
    out0 << chunk_data->ib_data.ip_type << "\n";
    uint32_t *data = (uint32_t *)get_ptr(addr, size * 4);
    if (data) {
      std::string cs_type = "unknown";
      if (chunk_data->ib_data.ip_type == AMDGPU_HW_IP_DMA)
        cs_type = "dma";
      else if ((chunk_data->ib_data.flags & 3) == 0)
        cs_type = "de";
      else if ((chunk_data->ib_data.flags & 3) == 1)
        cs_type = "ce";
      else if ((chunk_data->ib_data.flags & 3) == 3)
        cs_type = "ce_preamble";
      else
        fprintf(stderr, "%d %x\n", chunk_data->ib_data.ip_type, chunk_data->ib_data.flags);

      std::ofstream out(get_output_dir() + "cs." + std::to_string(cs_id) + "." +
                        cs_type + ".txt");
      out << std::hex << addr << std::dec << "\n";
      if (chunk_data->ib_data.ip_type == AMDGPU_HW_IP_DMA)
        process_dma_ib(out, data, data + size);
      else
        process_ib(out, data, data + size, registers);
    } else
      fprintf(stderr, "Failed to get IB data\n");
    ++cs_id;
#endif
  }
  return get_vtable().fn_cs_submit_raw2(device, context, bo_list_handle, num_chunks, chunks, seq_no);
}

int amdgpu_bo_alloc(amdgpu_device_handle dev,
                    struct amdgpu_bo_alloc_request *alloc_buffer,
                    amdgpu_bo_handle *buf_handle) {
  std::lock_guard<std::mutex> lock(global_mutex);

  auto ret = get_vtable().fn_bo_alloc(dev, alloc_buffer, buf_handle);
  if (ret) {
    return ret;
  }

  Buffer_info info = {};
  info.request = *alloc_buffer;
  buffers[*buf_handle] = info;
  return ret;
}

int amdgpu_bo_free(amdgpu_bo_handle buf_handle) {
  std::lock_guard<std::mutex> lock(global_mutex);

  auto it = buffers.find(buf_handle);
  if (it != buffers.end()) {
    if (it->second.data)
      get_vtable().fn_bo_cpu_unmap(buf_handle);
    buffers.erase(it);
  }

  return get_vtable().fn_bo_free(buf_handle);
}

int amdgpu_bo_cpu_map(amdgpu_bo_handle buf_handle, void **cpu) {
  std::lock_guard<std::mutex> lock(global_mutex);
  auto it = buffers.find(buf_handle);
  if (it != buffers.end() && it->second.data) {
    *cpu = it->second.data;
    return 0;
  }

  int ret = get_vtable().fn_bo_cpu_map(buf_handle, cpu);
  if (ret)
    return ret;

  if (it != buffers.end()) {
    it->second.data = *cpu;
  }
  return 0;
}

int amdgpu_bo_cpu_unmap(amdgpu_bo_handle buf_handle) {
  int ret = get_vtable().fn_bo_cpu_unmap(buf_handle);
  if (ret)
    return ret;

  std::lock_guard<std::mutex> lock(global_mutex);
  auto it = buffers.find(buf_handle);
  if (it != buffers.end()) {
    it->second.data = nullptr;
  }
  return 0;
}

int amdgpu_bo_va_op(amdgpu_bo_handle bo, uint64_t offset, uint64_t size,
                    uint64_t addr, uint64_t flags, uint32_t ops) {
  int ret = get_vtable().fn_bo_va_op(bo, offset, size, addr, flags, ops);
  if (ret)
    return ret;

  std::lock_guard<std::mutex> lock(global_mutex);
  if (ops == AMDGPU_VA_OP_MAP) {
    Map_info info;
    info.bo = bo;
    info.addr = addr;
    info.size = size;
    info.offset = offset;
    maps[addr] = info;
  } else if (ops == AMDGPU_VA_OP_UNMAP) {
    auto it = maps.find(addr);
    if (it != maps.end()) {
      maps.erase(it);
    }
  }
  return ret;
}

int amdgpu_bo_va_op_raw(amdgpu_device_handle dev, amdgpu_bo_handle bo, uint64_t offset, uint64_t size,
                    uint64_t addr, uint64_t flags, uint32_t ops) {
  int ret = get_vtable().fn_bo_va_op_raw(dev, bo, offset, size, addr, flags, ops);
  if (ret)
    return ret;

  std::lock_guard<std::mutex> lock(global_mutex);
  if (ops == AMDGPU_VA_OP_MAP) {
    Map_info info;
    info.bo = bo;
    info.addr = addr;
    info.size = size;
    info.offset = offset;
    maps[addr] = info;
  } else if (ops == AMDGPU_VA_OP_UNMAP) {
    auto it = maps.find(addr);
    if (it != maps.end()) {
      maps.erase(it);
    }
  }
  return ret;
}

/******************************************************************************
 * Forwarding stubs
 *****************************************************************************/

int amdgpu_cs_ctx_create2(amdgpu_device_handle dev,
                          uint32_t priority,
                          amdgpu_context_handle *context)
{
  return get_vtable().fn_cs_ctx_create2(dev, priority, context);
}

int amdgpu_cs_ctx_create(amdgpu_device_handle dev,
                         amdgpu_context_handle *context)
{
  return get_vtable().fn_cs_ctx_create(dev, context);
}

int amdgpu_cs_ctx_free(amdgpu_context_handle context)
{
  return get_vtable().fn_cs_ctx_free(context);
}

int amdgpu_cs_syncobj_import_sync_file(amdgpu_device_handle dev,
                                       uint32_t syncobj,
                                       int sync_file_fd)
{
  return get_vtable().fn_cs_syncobj_import_sync_file(dev, syncobj, sync_file_fd);
}

int amdgpu_cs_syncobj_import_sync_file2(amdgpu_device_handle dev,
                                        uint32_t syncobj,
                                        uint64_t point,
                                        int sync_file_fd)
{
  return get_vtable().fn_cs_syncobj_import_sync_file2(dev, syncobj, point, sync_file_fd);
}

int amdgpu_cs_query_reset_state(amdgpu_context_handle context,
                                uint32_t *state, uint32_t *hangs)

{
  return get_vtable().fn_cs_query_reset_state(context, state, hangs);
}

int amdgpu_va_range_alloc(amdgpu_device_handle dev,
                          enum amdgpu_gpu_va_range va_range_type,
                          uint64_t size,
                          uint64_t va_base_alignment,
                          uint64_t va_base_required,
                          uint64_t *va_base_allocated,
                          amdgpu_va_handle *va_range_handle,
                          uint64_t flags)
{
  return get_vtable().fn_va_range_alloc(dev, va_range_type, size,
                                        va_base_alignment, va_base_required,
                                        va_base_allocated, va_range_handle,
                                        flags);
}

int amdgpu_va_range_free(amdgpu_va_handle va_range_handle)
{
  return get_vtable().fn_va_range_free(va_range_handle);
}

int amdgpu_va_range_query(amdgpu_device_handle dev,
                          enum amdgpu_gpu_va_range type,
                          uint64_t *start,
                          uint64_t *end)
{
  return get_vtable().fn_va_range_query(dev, type, start, end);
}

int amdgpu_cs_signal_semaphore(amdgpu_context_handle ctx,
                               uint32_t ip_type,
                               uint32_t ip_instance,
                               uint32_t ring,
                               amdgpu_semaphore_handle sem)
{
  return get_vtable().fn_cs_signal_semaphore(ctx, ip_type, ip_instance, ring, sem);
}

int amdgpu_device_initialize(int fd,
                             uint32_t *major_version,
                             uint32_t *minor_version,
                             amdgpu_device_handle *device_handle)
{
  return get_vtable().fn_device_initialize(fd, major_version, minor_version, device_handle);
}

int amdgpu_device_deinitialize(amdgpu_device_handle device_handle)
{
  return get_vtable().fn_device_deinitialize(device_handle);
}

int amdgpu_bo_set_metadata(amdgpu_bo_handle buf_handle,
                           struct amdgpu_bo_metadata *info)
{
  return get_vtable().fn_bo_set_metadata(buf_handle, info);
}

const char *amdgpu_get_marketing_name(amdgpu_device_handle dev)
{
  return get_vtable().fn_get_marketing_name(dev);
}

int amdgpu_query_info(amdgpu_device_handle dev, unsigned info_id,
                      unsigned size, void *value)
{
  return get_vtable().fn_query_info(dev, info_id, size, value);
}

int amdgpu_query_hw_ip_count(amdgpu_device_handle dev, unsigned type,
                             uint32_t *count)
{
  return get_vtable().fn_query_hw_ip_count(dev, type, count);
}

int amdgpu_query_sw_info(amdgpu_device_handle dev, enum amdgpu_sw_info info,
                         void *value)
{
  return get_vtable().fn_query_sw_info(dev, info, value);
}

int amdgpu_query_hw_ip_info(amdgpu_device_handle dev, unsigned type,
                            unsigned ip_instance,
                            struct drm_amdgpu_info_hw_ip *info)
{
  return get_vtable().fn_query_hw_ip_info(dev, type, ip_instance, info);
}

int amdgpu_query_heap_info(amdgpu_device_handle dev, uint32_t heap,
                           uint32_t flags, struct amdgpu_heap_info *info)
{
  return get_vtable().fn_query_heap_info(dev, heap, flags, info);
}

int amdgpu_query_gds_info(amdgpu_device_handle dev,
                          struct amdgpu_gds_resource_info *gds_info)
{
  return get_vtable().fn_query_gds_info(dev, gds_info);
}

int amdgpu_query_gpu_info(amdgpu_device_handle dev,
                          struct amdgpu_gpu_info *info)
{
  return get_vtable().fn_query_gpu_info(dev, info);
}

int amdgpu_query_buffer_size_alignment(amdgpu_device_handle dev,
                                       struct amdgpu_buffer_size_alignments *info)
{
  return get_vtable().fn_query_buffer_size_alignment(dev, info);
}

int amdgpu_query_sensor_info(amdgpu_device_handle dev, unsigned sensor_type,
                             unsigned size, void *value)
{
  return get_vtable().fn_query_sensor_info(dev, sensor_type, size, value);
}

int amdgpu_bo_query_info(amdgpu_bo_handle buf_handle,
                         struct amdgpu_bo_info *info)
{
  return get_vtable().fn_bo_query_info(buf_handle, info);
}

int amdgpu_query_firmware_version(amdgpu_device_handle dev, unsigned fw_type,
                                  unsigned ip_instance, unsigned index,
                                  uint32_t *version, uint32_t *feature)
{
  return get_vtable().fn_query_firmware_version(dev, fw_type, ip_instance,
                                                index, version, feature);
}

int amdgpu_query_crtc_from_id(amdgpu_device_handle dev, unsigned id,
                              int32_t *result)
{
  return get_vtable().fn_query_crtc_from_id(dev, id, result);
}

int amdgpu_read_mm_registers(amdgpu_device_handle dev, unsigned dword_offset,
                             unsigned count, uint32_t instance, uint32_t flags,
                             uint32_t *values)
{
  return get_vtable().fn_read_mm_registers(dev, dword_offset, count, instance, flags, values);
}

int amdgpu_bo_list_create_raw(amdgpu_device_handle dev,
                              uint32_t number_of_buffers,
                              struct drm_amdgpu_bo_list_entry *buffers,
                              uint32_t *result)
{
  return get_vtable().fn_bo_list_create_raw(dev, number_of_buffers, buffers, result);
}

int amdgpu_bo_list_destroy_raw(amdgpu_device_handle dev, uint32_t bo_list)
{
  return get_vtable().fn_bo_list_destroy_raw(dev, bo_list);
}

int amdgpu_bo_list_create(amdgpu_device_handle dev,
                          uint32_t number_of_resources,
                          amdgpu_bo_handle *resources,
                          uint8_t *resource_prios,
                          amdgpu_bo_list_handle *result)
{
  int ret = get_vtable().fn_bo_list_create(dev, number_of_resources, resources, resource_prios, result);
  if (ret)
    return ret;

  fprintf(stderr, "bo list create %p with %u resources\n", *result, number_of_resources);

  {
    std::vector<std::pair<amdgpu_bo_handle, int>> handles;
    for (uint32_t i = 0; i < number_of_resources; ++i) {
      handles.push_back({resources[i], resource_prios ? resource_prios[i] : -1});
    }

    std::unique_lock<std::mutex> lock(global_mutex);
    bo_lists[*result] = std::move(handles);
  }
  return ret;
}

int amdgpu_bo_list_destroy(amdgpu_bo_list_handle handle)
{
  {
    std::unique_lock<std::mutex> lock(global_mutex);
    auto it = bo_lists.find(handle);
    if (it != bo_lists.end())
      bo_lists.erase(it);
  }
  fprintf(stderr, "bo list destroy %p\n", handle);
  return get_vtable().fn_bo_list_destroy(handle);
}

int amdgpu_bo_list_update(amdgpu_bo_list_handle handle,
                          uint32_t number_of_resources,
                          amdgpu_bo_handle *resources,
                          uint8_t *resource_prios)
{
  return get_vtable().fn_bo_list_update(handle, number_of_resources, resources,
                                        resource_prios);
}

int amdgpu_vm_reserve_vmid(amdgpu_device_handle dev, uint32_t flags)
{
  return get_vtable().fn_vm_reserve_vmid(dev, flags);
}

int amdgpu_vm_unreserve_vmid(amdgpu_device_handle dev, uint32_t flags)
{
  return get_vtable().fn_vm_unreserve_vmid(dev, flags);
}

int amdgpu_bo_import(amdgpu_device_handle dev,
                     enum amdgpu_bo_handle_type type,
                     uint32_t shared_handle,
                     struct amdgpu_bo_import_result *output)
{
  return get_vtable().fn_bo_import(dev, type, shared_handle, output);
}

int amdgpu_bo_export(amdgpu_bo_handle buf_handle,
                     enum amdgpu_bo_handle_type type,
                     uint32_t *shared_handle)
{
  {
    std::unique_lock<std::mutex> lock(global_mutex);
    auto it = buffers.find(buf_handle);
    switch(type) {
    case amdgpu_bo_handle_type_dma_buf_fd:
      it->second.request.flags |= 0x1000;
    }
  }
  return get_vtable().fn_bo_export(buf_handle, type, shared_handle);
}

int amdgpu_create_bo_from_user_mem(amdgpu_device_handle dev,
                                   void *cpu, uint64_t size,
                                   amdgpu_bo_handle *buf_handle)
{
  return get_vtable().fn_create_bo_from_user_mem(dev, cpu, size, buf_handle);
}

void amdgpu_bo_inc_ref(amdgpu_bo_handle bo)
{
  return get_vtable().fn_bo_inc_ref(bo);
}

int amdgpu_find_bo_by_cpu_mapping(amdgpu_device_handle dev,
                                  void *cpu,
                                  uint64_t size,
                                  amdgpu_bo_handle *buf_handle,
                                  uint64_t *offset_in_bo)
{
  return get_vtable().fn_find_bo_by_cpu_mapping(dev, cpu, size, buf_handle,
                                                offset_in_bo);
}

int amdgpu_cs_create_semaphore(amdgpu_semaphore_handle *sem)
{
  return get_vtable().fn_cs_create_semaphore(sem);
}

int amdgpu_cs_wait_semaphore(amdgpu_context_handle ctx,
                             uint32_t ip_type,
                             uint32_t ip_instance,
                             uint32_t ring,
                             amdgpu_semaphore_handle sem)
{
  return get_vtable().fn_cs_wait_semaphore(ctx, ip_type, ip_instance, ring, sem);
}

int amdgpu_cs_destroy_semaphore(amdgpu_semaphore_handle sem)
{
  return get_vtable().fn_cs_destroy_semaphore(sem);
}


void amdgpu_cs_chunk_fence_to_dep(struct amdgpu_cs_fence *fence,
                                  struct drm_amdgpu_cs_chunk_dep *dep)
{
  return get_vtable().fn_cs_chunk_fence_to_dep(fence, dep);
}

void amdgpu_cs_chunk_fence_info_to_data(struct amdgpu_cs_fence_info *fence_info,
                                        struct drm_amdgpu_cs_chunk_data *data)
{
  return get_vtable().fn_cs_chunk_fence_info_to_data(fence_info, data);
}

int amdgpu_bo_wait_for_idle(amdgpu_bo_handle buf_handle,
                            uint64_t timeout_ns,
                            bool *buffer_busy)
{
  return get_vtable().fn_bo_wait_for_idle(buf_handle, timeout_ns, buffer_busy);
}

int amdgpu_cs_wait_fences(struct amdgpu_cs_fence *fences,
                          uint32_t fence_count,
                          bool wait_all,
                          uint64_t timeout_ns,
                          uint32_t *status, uint32_t *first)
{
  return get_vtable().fn_cs_wait_fences(fences, fence_count, wait_all,
                                        timeout_ns, status, first);
}

int amdgpu_cs_fence_to_handle(amdgpu_device_handle dev,
                              struct amdgpu_cs_fence *fence,
                              uint32_t what,
                              uint32_t *out_handle)
{
  return get_vtable().fn_cs_fence_to_handle(dev, fence, what, out_handle);
}

int amdgpu_cs_query_fence_status(struct amdgpu_cs_fence *fence,
                                 uint64_t timeout_ns,
                                 uint64_t flags,
                                 uint32_t *expired)
{
  return get_vtable().fn_cs_query_fence_status(fence, timeout_ns, flags,
                                               expired);
}

int amdgpu_cs_query_reset_state2(amdgpu_context_handle context,
                                 uint64_t *flags)
{
  return get_vtable().fn_cs_query_reset_state2(context, flags);
}

int amdgpu_cs_syncobj_export_sync_file(amdgpu_device_handle dev,
                                       uint32_t syncobj,
                                       int *sync_file_fd)
{
  return get_vtable().fn_cs_syncobj_export_sync_file(dev, syncobj,
                                                     sync_file_fd);
}

int amdgpu_cs_syncobj_export_sync_file2(amdgpu_device_handle dev,
                                        uint32_t syncobj,
                                        uint64_t point,
                                        uint32_t flags,
                                        int *sync_file_fd)
{
  return get_vtable().fn_cs_syncobj_export_sync_file2(dev, syncobj, point,
                                                      flags, sync_file_fd);
}

int amdgpu_cs_create_syncobj(amdgpu_device_handle dev,
                             uint32_t *syncobj)
{
  return get_vtable().fn_cs_create_syncobj(dev, syncobj);
}

int amdgpu_cs_create_syncobj2(amdgpu_device_handle dev,
                              uint32_t  flags,
                              uint32_t *syncobj)
{
  return get_vtable().fn_cs_create_syncobj2(dev, flags, syncobj);
}

int amdgpu_cs_destroy_syncobj(amdgpu_device_handle dev,
                              uint32_t syncobj)
{
  return get_vtable().fn_cs_destroy_syncobj(dev, syncobj);
}

int amdgpu_cs_ctx_override_priority(amdgpu_device_handle dev,
                                    amdgpu_context_handle context,
                                    int master_fd,
                                    unsigned priority)
{
  return get_vtable().fn_cs_ctx_override_priority(dev, context, master_fd,
                                                  priority);
}

int amdgpu_cs_syncobj_reset(amdgpu_device_handle dev,
                            const uint32_t *syncobjs, uint32_t syncobj_count)
{
  return get_vtable().fn_cs_syncobj_reset(dev, syncobjs, syncobj_count);
}

int amdgpu_cs_syncobj_wait(amdgpu_device_handle dev,
                           uint32_t *handles, unsigned num_handles,
                           int64_t timeout_nsec, unsigned flags,
                           uint32_t *first_signaled)
{
  return get_vtable().fn_cs_syncobj_wait(dev, handles, num_handles,
                                         timeout_nsec, flags, first_signaled);
}

int amdgpu_cs_syncobj_query(amdgpu_device_handle dev,
                            uint32_t *handles, uint64_t *points,
                            unsigned num_handles)
{
  return get_vtable().fn_cs_syncobj_query(dev, handles, points, num_handles);
}

int amdgpu_cs_syncobj_query2(amdgpu_device_handle dev,
                             uint32_t *handles, uint64_t *points,
                             unsigned num_handles, uint32_t flags)
{
  return get_vtable().fn_cs_syncobj_query2(dev, handles, points, num_handles,
                                           flags);
}

int amdgpu_cs_syncobj_transfer(amdgpu_device_handle dev,
                               uint32_t dst_handle,
                               uint64_t dst_point,
                               uint32_t src_handle,
                               uint64_t src_point,
                               uint32_t flags)
{
  return get_vtable().fn_cs_syncobj_transfer(dev, dst_handle, dst_point,
                                             src_handle, src_point, flags);
}

int amdgpu_cs_syncobj_timeline_wait(amdgpu_device_handle dev,
                                    uint32_t *handles, uint64_t *points,
                                    unsigned num_handles,
                                    int64_t timeout_nsec, unsigned flags,
                                    uint32_t *first_signaled)
{
  return get_vtable().fn_cs_syncobj_timeline_wait(dev, handles, points,
                                                  num_handles, timeout_nsec,
                                                  flags, first_signaled);
}

int amdgpu_cs_syncobj_timeline_signal(amdgpu_device_handle dev,
                                      const uint32_t *syncobjs,
                                      uint64_t *points,
                                      uint32_t syncobj_count)
{
  return get_vtable().fn_cs_syncobj_timeline_signal(dev, syncobjs, points,
                                                    syncobj_count);
}

int amdgpu_cs_syncobj_signal(amdgpu_device_handle dev,
                             const uint32_t *syncobjs, uint32_t syncobj_count)
{
  return get_vtable().fn_cs_syncobj_signal(dev, syncobjs, syncobj_count);
}

int amdgpu_cs_import_syncobj(amdgpu_device_handle dev,
                             int shared_fd,
                             uint32_t *syncobj)
{
  return get_vtable().fn_cs_import_syncobj(dev, shared_fd, syncobj);
}

int amdgpu_cs_export_syncobj(amdgpu_device_handle dev,
                             uint32_t syncobj,
                             int *shared_fd)
{
  return get_vtable().fn_cs_export_syncobj(dev, syncobj, shared_fd);
}

extern "C" void *dlopen(const char *filename, int flags)
{
  static void *(*real_dlopen)(const char *, int);
  if (!real_dlopen) {
    real_dlopen = (void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen");
  }
  if (strcmp(filename, "libdrm_amdgpu.so.1") == 0) {
    Dl_info info;
    dladdr((void*)&dlopen, &info);
    return real_dlopen(info.dli_fname, flags);
  }

  return real_dlopen(filename, flags);
}

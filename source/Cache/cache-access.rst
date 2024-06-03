.. _Cache-Access:

Cache的访问
============

Cache作为一种高速存储资源，通过存储近期或频繁访问的数据，极大地减少了处理器与主存之间的数据传输时间，从而提升了系统的整体性能。本章主要介绍 GPGPU-Sim 如何模拟对 Cache 的访问过程。

.. _Cache-Basic-Knowledge:

Cache 基础
-------------------

首先需要了解的就是 `Sector Cache` 和 `Line Cache` 的 :ref:`Cache-Block-Org`、 :ref:`Cache-Block-Addrmap` 以及 :ref:`Cache-Block-Status`。

.. _Cache-Block-Org:

Cache Block 的组织方式
++++++++++++++++++++++

- `Line Cache`：Line Cache是最常见的缓存组织方式之一，它按固定大小的cache block来存储数据。这些行的大小取决于具体的GPU型号，在模拟器中的  :ref:`Cache-Config` 中进行配置。当LDST单元尝试读取或写入数据时，整个cache block（包含所需数据的那部分）被加载到缓存中。因为程序往往具有良好的空间局部性，即接近已访问数据的其他数据很可能很快会被访问，行缓存可以有效利用这一点，提高缓存命中率，从而提高整体性能。

- `Sector Cache`：与Line Cache相比，Sector Cache提供了一种更为灵活的缓存数据方式。在Sector Cache中，每个cache block被进一步细分为若干个sector或称为“扇区”。这样，当请求特定数据时，只有包含这些数据的特定扇区会被加载到缓存中，而不是整个cache block。这种方法在数据的空间局部性不是非常理想的情况下尤其有效，因为它减少了不必要数据的缓存，从而为其他数据的缓存留出了空间，提高了缓存的有效性。

下面的代码块 :ref:`fig-cache-block-org` 用一个 `4`-sets/`6`-ways 的简单Cache展示了Cache Block的两种组织形式。

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Cache Block的组织形式
  :name: fig-cache-block-org

  // 1. 如果是 Line Cache：
  //  4 sets, 6 ways, cache blocks are not split to sectors.
  //  |-----------------------------------|
  //  |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
  //  |-----------------------------------|
  //  |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
  //  |-----------------------------------|
  //  |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
  //  |-----------------------------------|
  //  |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
  //  |--------------|--------------------|
  //                 |--------> 20 is the cache block index
  // 2. 如果是 Sector Cache：
  //  4 sets, 6 ways, each cache block is split to 4 sectors.
  //  |-----------------------------------|
  //  |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
  //  |-----------------------------------|
  //  |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
  //  |-----------------------------------|
  //  |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
  //  |-----------------------------------|
  //  |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
  //  |-----------/-----\-----------------|
  //             /       \
  //            /         \--------> 20 is the cache block index
  //           /           \
  //          /             \
  //         |---------------|
  //         | 3 | 2 | 1 | 0 | // a block contains 4 sectors
  //         |---------|-----|
  //                   |--------> 1 is the sector offset in the 20-th cache block

.. _Cache-Block-Addrmap:

访存地址的映射
++++++++++++++++++

本节主要介绍一个访存地址如何映射到一个 cache block。如何将一个特定的访存地址映射到这些 cache block 上，是通过一系列精确计算的步骤完成的。其中最关键的两步是计算 `Tag 位`` 和 `Set Index 位`。 `Tag 位` 用于标识数据存储在哪个缓存块中，而 `Set Index 位` 则用于确定这个地址映射到缓存中的哪一个 Set 上。不过，即使能够确定一个地址映射到了哪个 Set，要找到这个数据是否已经被缓存（以及具体存储在哪一路），还需进行遍历查找，这一步是判断缓存命中还是缓存失效的关键环节。接下来，将详细介绍 `Tag 位` 和 `Set Index 位` 的计算过程，这两者是理解地址映射机制的重点。

Tag 位的计算
^^^^^^^^^^^^^^^^^

下面的代码块 :ref:`fig-tag-calc` 展示了如何由访存地址 `addr` 计算 `tag 位`。

这里需要注意的是，最新版本中的 GPGPU-Sim 中的 `tag 位` 是由 `index 位` 和 `traditional tag 位` 共同组成的（这里所说的 `traditional tag 位` 就是指传统 CPU 上 Cache 的 `tag 位` 的计算方式： ``traditional tag = addr >> (log2(m_line_sz) + log2(m_nset))``，详见 :ref:`fig-cache-block-addrmap` 示意图），其中 `m_line_sz` 和 `m_nset` 分别是 Cache 的 `line size` 和 `set` 的数量），这样可以允许更复杂的 `set index 位` 的计算，从而避免将 `set index 位` 不同但是 `traditional tag 位` 相同的地址映射到同一个 `set`。这里是把完整的 [`traditional tag 位 + set index 位 + log2(m_line_sz)'b0`] 来作为 `tag 位`。


.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Tag 位的计算
  :name: fig-tag-calc

  typedef unsigned long long new_addr_type;

  // m_line_sz：cache block的大小，单位是字节。
  new_addr_type tag(new_addr_type addr) const {
    // For generality, the tag includes both index and tag. This allows for more
    // complex set index calculations that can result in different indexes
    // mapping to the same set, thus the full tag + index is required to check
    // for hit/miss. Tag is now identical to the block address.
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }

Block Address 的计算
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Block Address 的计算
  :name: fig-block-addr-calc

  // m_line_sz：cache block的大小，单位是字节。
  new_addr_type block_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }

Block Address 的计算与 Tag 位的计算是一样的，都是通过 `m_line_sz` 来计算的。`block_addr` 函数会返回一个地址 `addr` 在 Cache 中的 `block address`，这里是把完整的 [`traditional tag 位 + set index 位 + log2(m_line_sz)'b0`] 来作为 `tag 位`。

Set Index 位的计算
^^^^^^^^^^^^^^^^^^^^^^^^

GPGPU-Sim 中真正实现的 `set index 位` 的计算方式是通过 `cache_config::set_index()` 和 `l2_cache_config::set_index()` 函数来实现的，这个函数会返回一个地址 `addr` 在 Cache 中的 `set index`。这里的 `set index` 有一整套的映射函数，尤其是 L2 Cache 的映射方法十分复杂（涉及到内存子分区的概念），这里先不展开讨论。对于 L2 Cache 暂时只需要知道， `set_index()` 函数会计算并返回一个地址 `addr` 在 Cache 中的 `set index`，具体如何映射后续再讲。

这里仅介绍一下 GV100 架构中的 L1D Cache 的 `set index 位` 的计算方式，如 :ref:`fig-set-index-calc` 所示：

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: GV100 架构中的 L1D Cache 的 Set Index 位的计算
  :name: fig-set-index-calc

  // m_nset：cache set 的数量。
  // m_line_sz_log2：cache block 的大小的对数。
  // m_nset_log2：cache set 的数量的对数。
  // m_index_function：set index 的计算函数，GV100 架构中的 L1D Cache 的配置为 
  //                   LINEAR_SET_FUNCTION。
  unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                       unsigned m_line_sz_log2,
                                       unsigned m_nset_log2,
                                       unsigned m_index_function) const {
    unsigned set_index = 0;
    switch (m_index_function) {
      // ......
      case LINEAR_SET_FUNCTION: {
        // addr: [m_line_sz_log2-1:0]                          => byte offset
        // addr: [m_line_sz_log2+m_nset_log2-1:m_line_sz_log2] => set index
        set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
        break;
      }
      default: {
        assert("\nUndefined set index function.\n" && 0);
        break;
      }
    }
    assert((set_index < m_nset) &&
           "\nError: Set index out of bounds. This is caused by "
           "an incorrect or unimplemented custom set index function.\n");
    return set_index;
  }

.. NOTE::

  `Set Index 位` 有一整套的映射函数，这里只是简单介绍了 GV100 架构中的 L1D Cache 的 `Set Index 位` 的计算结果，具体的映射函数会在后续章节中详细介绍。


访问地址的映射示意图
^^^^^^^^^^^^^^^^^^^^^^^^^^

下面的代码块 :ref:`fig-cache-block-addrmap` 用一个访存地址 `addr` 展示了访问 Cache 的地址映射。

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Cache Block的地址映射
  :name: fig-cache-block-addrmap
  
  // 1. 如果是 Line Cache：
  //  MSHR 的地址即为地址 addr 的 [tag 位 + set_index 位]。即除 offset in-line 
  //  位以外的所有位。
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //  |<-Tr-->|            \                     /
  //  |-------|-------------|-------------------| // addr
  //             set_index     offset in-line
  //  |<--------tag--------> 0 0 0 0 0 0 0 0 0 0|
  // 2. 如果是 Sector Cache：
  //  MSHR 的地址即为地址 addr 的 [tag 位 + set_index 位 + sector offset 位]。
  //  即除 offset in-sector 位以外的所有位。
  //  |<----------mshr_addr----------->|
  //                     sector offset  offset in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //  |<-Tr-->|            \                     /
  //  |-------|-------------|-------------------| // addr
  //             set_index     offset in-line
  //  |<--------tag--------> 0 0 0 0 0 0 0 0 0 0|


.. hint::
  
  :ref:`fig-cache-block-addrmap` 中所展示的是最新版本 GPGPU-Sim 的实现， `tag 位` 是由 `index 位` 和 `traditional tag 位` 共同组成的。 `traditional tag 位` 如图中 `Tr` 的范围所示。


.. _Cache-Block-Status:

Cache Block的状态
+++++++++++++++++++++++

Cache Block 的状态是指在 Line Cache 中 cache block 或者在 Sector Cache 中的 cache sector 的状态，包含以下几种：

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Cache Block State
  :name: code-cache-block-status

  enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };

这里会区分开 Line Cache 和 Sector Cache 的状态介绍，因为 Line Cache 和 Sector Cache 的状态是不同的。对于 Line Cache 来说，每个 ``line_cache_block *line`` 对象都有一个标识状态的成员变量 ``m_status``，它的值是 ``enum cache_block_state`` 中的一种。具体的状态如下：

- ``INVALID``: cache block 无效数据。需要注意的是，这里的无效与下面的 ``MODIFIED`` 和 ``RESERVED`` 不同，意味着当前 cache block 没有存储任何有效数据。
- ``VALID``: 当一个 cache block 的状态是 ``VALID``，说明该 block 的数据是有效的，可以为 cache 提供命中的数据。
- ``MODIFIED``: 如果 cache block 状态是 ``MODIFIED``，说明该 block 的数据已经被其他线程修改。如果当前访问也是写操作的话即为命中；但如果不是写操作，则需要判断当前 cache block 是否已被修改完毕并可读（由 ``bool m_readable`` 确定），修改完毕并可读的话（``m_readable = true``）则为命中，不可读的话（``m_readable = false``）则发生  ``SECTOR_MISS``。
- ``RESERVED``: 当一个 cache block 被分配以重新填充未命中的数据，即需要装入新的数据以应对未命中（``MISS``）情况时（如果一次数据访问 cache 或者一个数据填充进 cache 时发生 ``MISS``），cache block 的状态 ``m_status`` 被设置为 ``RESERVED``，这意味着该 block 正在准备或已经准备好重新填充新的数据。

而对于 Sector Cache 来说，每个 ``sector_cache_block_t *line`` 对象都有一个 ``cache_block_state *m_status`` 数组。数组的大小是 ``const unsigned SECTOR_CHUNCK_SIZE = 4`` 即每个 cache line 都有 `4` 个 sector，这个状态数组用以标识每个 sector 的状态，它的每一个元素也都是 ``cache_block_state`` 中的一个。具体的状态与上述 Line Cache 中的状态的唯一区别就是，``cache_block_state *m_status`` 数组的每个元素标识每个 sector 的状态，而不是 Line Cache 中的整个 cache block 的状态。

Cache 的访问状态
------------------------

Cache 的访问状态有以下几种：

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Cache Request Status
  :name: code-cache-request-status

  enum cache_request_status {
    // 命中。
    HIT = 0,
    // 保留成功，当访问地址被映射到一个已经被分配的 cache block/sector 时，block/
    // sector 的状态被设置为 RESERVED。
    HIT_RESERVED,
    // 未命中。
    MISS,
    // 保留失败。
    RESERVATION_FAIL,
    // Sector缺失。
    SECTOR_MISS,
    MSHR_HIT,
    // cache_request_status的状态总数。
    NUM_CACHE_REQUEST_STATUS
  };

Cache 的组织和实现
------------------------

TODO









Cache 的访问状态
------------------------

在访问 Cache 的时候，会调用 ``access()`` 函数，例如 ``m_L2cache->access()``，``m_L1I->access()``，``m_L1D->access()`` 等。

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Data Cache Access Function
  :name: code-cache-access-function

  enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                               unsigned time,
                                               std::list<cache_event> &events) {
    assert(mf->get_data_size() <= m_config.get_atom_sz());
    bool wr = mf->get_is_write();
    // Block Address 的计算与 Tag 位的计算是一样的，都是通过 m_line_sz 来计算的。
    // block_addr 函数会返回一个地址 addr 在 Cache 中的 block address，这里是把
    // 完整的 [traditional tag 位 + set index 位 + log2(m_line_sz)'b0] 来作为 
    // tag 位。
    new_addr_type block_addr = m_config.block_addr(addr);

    unsigned cache_index = (unsigned)-1;
    // 判断对 cache 的访问（地址为 addr）是 HIT / HIT_RESERVED / SECTOR_MISS / 
    // MISS / RESERVATION_FAIL 等状态。且如果返回的 cache 访问为 MISS，则将需要
    // 被替换的 cache block 的索引写入 cache_index。
    enum cache_request_status probe_status =
        m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
    // 主要包括上述各种对 cache 的访问的状态下的执行对 cache 访问的操作，例如：
    //   (this->*m_wr_hit)、(this->*m_wr_miss)、
    //   (this->*m_rd_hit)、(this->*m_rd_miss)。
    enum cache_request_status access_status =
        process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
    m_stats.inc_stats(mf->get_access_type(),
                      m_stats.select_stats_status(probe_status, access_status));
    m_stats.inc_stats_pw(mf->get_access_type(), m_stats.select_stats_status(
                                                    probe_status, access_status));
    return access_status;
  }

.. 然后 Cache 会调用 ``tag_array::probe()`` 函数来判断 

.. 接下来将首先介绍 ``tag_array::probe()`` 函数的实现，然后再介绍 Cache 的访问状态有哪些。



.. - ``HIT_RESERVED`` ：对于Sector Cache来说，如果Cache block[mask]状态是RESERVED，说明有其他的线程正在读取这个Cache block。挂起的命中访问已命中处于RESERVED状态的缓存行，这意味着同一行上已存在由先前缓存未命中发送的flying内存请求。
.. - **HIT** ：
.. - **SECTOR_MISS** ：
.. - **RESERVATION_FAIL** ：
.. - **MISS** ：
.. - **MSHR_HIT** ：



.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: tag_array::probe() 函数
  :name: code-cache-tag_array-probe
  

  enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                             mem_access_sector_mask_t mask,
                                             bool is_write, bool probe_mode,
                                             mem_fetch *mf) const {
    // 这里的输入地址 addr 是 cache block 的地址，该地址即为由 block_addr() 计算
    // 而来。
    // m_config.set_index(addr) 是返回一个地址 addr 在 Cache 中的 set index。这
    // 里的 set index 有一整套的映射函数。
    unsigned set_index = m_config.set_index(addr);
    // |-------|-------------|--------------|
    //            set_index   offset in-line
    // |<--------tag--------> 0 0 0 0 0 0 0 |
    // 这里实际返回的是 {除 offset in-line 以外的所有位, offset in-line'b0}，即 
    // set index 也作为 tag 位的一部分了。
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned long long valid_timestamp = (unsigned)-1;

    bool all_reserved = true;

    // check for hit or pending hit
    // 对所有的 Cache Ways 检查。需要注意这里其实是针对一个 set 的所有 way 进行检
    // 查，因为给定一个地址，可以确定它所在的 set index，然后再通过 tag 位 来匹配
    // 并确定这个地址在哪一个 way 上。
    for (unsigned way = 0; way < m_config.m_assoc; way++) {
      // For example, 4 sets, 6 ways:
      // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
      // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
      // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
      // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
      //                |--------> index => cache_block_t *line
      // index 是 cache block 的索引。
      unsigned index = set_index * m_config.m_assoc + way;
      cache_block_t *line = m_lines[index];
      // tag 位 相符，说明当前 cache block 已经是 addr 地址映射在当前 way 上。
      if (line->m_tag == tag) {
        // cache block 的状态，包含：
        //     enum cache_block_state { 
        //       INVALID = 0, RESERVED, VALID, MODIFIED };
        if (line->get_status(mask) == RESERVED) {
          // 当访问地址被映射到一个已经被分配的 cache block 或 cache sector 时，
          // cache block 或 cache sector 的状态被设置为 RESERVED。这说明当前 
          // block / sector 被分配给了其他的线程，而且正在读取的内容正是访问地址
          // addr 想要的数据。
          idx = index;
          return HIT_RESERVED;
        } else if (line->get_status(mask) == VALID) {
          // 如果 cache block 或 cache sector 的状态是 VALID，说明已经命中。
          idx = index;
          return HIT;
        } else if (line->get_status(mask) == MODIFIED) {
          // 如果 cache block 或 cache sector 的状态是 MODIFIED，说明该 block 
          // 或 sector 的数据已经被其他线程修改。如果当前访问也是写操作的话即为命
          // 中；但如果不是写操作，则需要判断当前 cache block 或 cache sector 
          // 是否已被修改完毕并可读（由 ``bool m_readable`` 确定），修改完毕并可
          // 读的话（``m_readable = true``）则为命中，不可读的话（``m_readable 
          // = false``）则发生 SECTOR_MISS。
          if ((!is_write && line->is_readable(mask)) || is_write) {
            idx = index;
            return HIT;
          } else {
            // for condition: is_write && line->is_readable(mask) == false.
            idx = index;
            return SECTOR_MISS;
          }
        } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
          // line cache 不会走这个分支，在 line cache 中，line->is_valid_line() 
          // 返回的是 m_status 的值，当其为 VALID 时，line->get_status(mask) 也
          // 是返回的 m_status 的值，即为 VALID，因此对于 line cache 这条分支无
          // 效。但是对于sector cache， 有：
          //   virtual bool is_valid_line() { return !(is_invalid_line()); }
          // 而 sector cache 中的 is_invalid_line() 是，只要有一个 sector 不为 
          // INVALID 即返回 false，因此 is_valid_line() 返回的是，只要存在一个 
          // sector 不为 INVALID 就设置 is_valid_line() 为真。所以这条分支对于 
          // sector cache 是可走的。
          // cache block 有效，但是其中的 byte mask = cache block[mask] 状态无
          // 效，说明 sector 缺失。
          idx = index;
          return SECTOR_MISS;
        } else {
          assert(line->get_status(mask) == INVALID);
        }
      }
      
      // 每一次循环中能走到这里的，即为当前 cache block 的 line->m_tag != tag。
      // 那么就需要考虑当前这 cache block 能否被逐出替换，请注意，这个判断是在对
      // 每一个 way 循环的过程中进行的，也就是说，假如第一个 cache block 没有返
      // 回以上访问状态，但有可能直到所有 way 的最后一个 cache block 才满足
      // m_tag != tag，但是在对第 0 ~ way-2 号的 cache block 循环判断的时候，
      // 就需要记录下每一个 way 的 cache block 是否能够被逐出。因为如果等到所有 
      // way 的 cache block 都没有满足 line->m_tag != tag 时，再回过头来循环所
      // 有 way 找最优先被逐出的 cache block 那就增加了模拟的开销。因此实际上对
      // 于所有 way 中的每一个 cache block，只要它不满足 line->m_tag != tag，
      // 就在这里判断它能否被逐出。
      // line->is_reserved_line()：只要有一个 sector 是 RESERVED，就认为这个 
      // Cache Line 是 RESERVED。这里即整个 line 没有 sector 是 RESERVED。
      if (!line->is_reserved_line()) {
        // percentage of dirty lines in the cache
        // number of dirty lines / total lines in the cache
        float dirty_line_percentage =
            ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
        // If the cacheline is from a load op (not modified), 
        // or the total dirty cacheline is above a specific value,
        // Then this cacheline is eligible to be considered for replacement 
        // candidate, i.e. Only evict clean cachelines until total dirty 
        // cachelines reach the limit.
        // m_config.m_wr_percent 在 V100 中配置为 25%。
        // line->is_modified_line()：只要有一个 sector 是 MODIFIED，就认为这
        // 个 cache line 是MODIFIED。这里即整个 line 没有 sector 是 MODIFIED，
        // 或者 dirty_line_percentage 超过了 m_config.m_wr_percent。
        if (!line->is_modified_line() ||
            dirty_line_percentage >= m_config.m_wr_percent) 
        {
          // 因为在逐出一个 cache block 时，优先逐出一个干净的块，即没有 sector 
          // 被 RESERVED，也没有 sector 被 MODIFIED，来逐出；但是如果 dirty 的
          // cache line 的比例超过 m_wr_percent（V100 中配置为 25%），也可以不
          // 满足 MODIFIED 的条件。
          // 在缓存管理机制中，优先逐出未被修改（"干净"）的缓存块的策略，是基于几
          // 个重要的考虑：
          // 1. 减少写回成本：缓存中的数据通常来源于更低速的后端存储（如主存储器）。
          //    当缓存块被修改（即包含"脏"数据）时，在逐出这些块之前，需要将这些更
          //    改写回到后端存储以确保数据一致性。相比之下，未被修改（"干净"）的缓
          //    存块可以直接被逐出，因为它们的内容已经与后端存储一致，无需进行写回
          //    操作。这样就避免了写回操作带来的时间和能量开销。
          // 2. 提高效率：写回操作相对于读取操作来说，是一个成本较高的过程，不仅涉
          //    及更多的时间延迟，还可能占用宝贵的带宽，影响系统的整体性能。通过先
          //    逐出那些"干净"的块，系统能够在维持数据一致性的前提下，减少对后端存
          //    储带宽的需求和写回操作的开销。
          // 3. 优化性能：选择逐出"干净"的缓存块还有助于维护缓存的高命中率。理想情
          //    况下，缓存应当存储访问频率高且最近被访问的数据。逐出"脏"数据意味着
          //    这些数据需要被写回，这个过程不仅耗时而且可能导致缓存暂时无法服务其
          //    他请求，从而降低缓存效率。
          // 4. 数据安全与完整性：在某些情况下，"脏"缓存块可能表示正在进行的写操作
          //    或者重要的数据更新。通过优先逐出"干净"的缓存块，可以降低因为缓存逐
          //    出导致的数据丢失或者完整性破坏的风险。
          
          // all_reserved 被初始化为 true，是指所有 cache line 都没有能够逐出来
          // 为新访问提供 RESERVE 的空间，这里一旦满足上面两个 if 条件，说明当前 
          // line 可以被逐出来提供空间供 RESERVE 新访问，这里 all_reserved 置为 
          // false。而一旦最终 all_reserved 仍旧保持 true 的话，就说明当前 set 
          // 里没有哪一个 way 的 cache block 可以被逐出，发生 RESERVATION_FAIL。
          all_reserved = false;
          // line->is_invalid_line() 标识所有 sector 都无效。
          if (line->is_invalid_line()) {
            // 尽管配置有 LRU 或者 FIFO 替换策略，但是最理想的情况还是优先替换整个 
            // cache block 都无效的块。因为这种无效的块不需要写回，能够节省带宽。
            invalid_line = index;
          } else {
            // valid_line aims to keep track of most appropriate replacement 
            // candidate.
            if (m_config.m_replacement_policy == LRU) {
              // valid_timestamp 设置为最近最少被使用的 cache line 的最末次访问
              // 时间。
              // valid_timestamp 被初始化为 (unsigned)-1，即可以看作无穷大。
              if (line->get_last_access_time() < valid_timestamp) {
                // 这里的 valid_timestamp 是周期数，即最小的周期数具有最大的被逐
                // 出优先级，当然这个变量在这里只是找具有最小周期数的 cache block，
                // 最小周期数意味着离他上次使用才最早，真正标识哪个 cache block 
                // 具有最大优先级被逐出的是valid_line。
                valid_timestamp = line->get_last_access_time();
                // 标识当前 cache block 具有最小的执行周期数，index 这个 cache 
                // block 应该最先被逐出。
                valid_line = index;
              }
            } else if (m_config.m_replacement_policy == FIFO) {
              if (line->get_alloc_time() < valid_timestamp) {
                // FIFO 按照最早分配时间的 cache block 最优先被逐出的原则。
                valid_timestamp = line->get_alloc_time();
                valid_line = index;
              }
            }
          }
        }
      } // 这里是把当前 set 里所有的 way 都循环一遍，如果找到了 line->m_tag == 
        // tag 的块，则已经返回了访问状态，如果没有找到，则也遍历了一遍所有 way 的
        // cache block，找到了最优先应该被逐出和替换的 cache block。
    }
    // all_reserved 被初始化为 true，是指所有 cache line 都没有能够逐出来为新访
    // 问提供 RESERVE 的空间，这里一旦满足上面两个 if 条件，说明当前 line 可以被
    // 逐出来提供空间供 RESERVE 新访问，这里 all_reserved 置为 false。而一旦最终 
    // all_reserved 仍旧保持 true 的话，就说明当前 set 里没有哪一个 way 的 cache 
    // block 可以被逐出，发生 RESERVATION_FAIL。
    if (all_reserved) {
      // all of the blocks in the current set have no enough space in cache 
      // to allocate on miss.
      assert(m_config.m_alloc_policy == ON_MISS);
      return RESERVATION_FAIL;  // miss and not enough space in cache to 
                                // allocate on miss
    }

    // 如果上面的 all_reserved 为 false，才会到这一步，即 cache line 可以被逐出
    // 来为新访问提供 RESERVE。
    if (invalid_line != (unsigned)-1) {
      // 尽管配置有 LRU 或者 FIFO 替换策略，但是最理想的情况还是优先替换整个 cache 
      // block 都无效的块。因为这种无效的块不需要写回，能够节省带宽。
      idx = invalid_line;
    } else if (valid_line != (unsigned)-1) {
      // 没有无效的块，就只能将上面按照 LRU 或者 FIFO 确定的 cache block 作为被
      // 逐出的块了。
      idx = valid_line;
    } else
      abort();  // if an unreserved block exists, it is either invalid or
                // replaceable

    // if (probe_mode && m_config.is_streaming()) {
    //   line_table::const_iterator i =
    //       pending_lines.find(m_config.block_addr(addr));
    //   assert(mf);
    //   if (!mf->is_write() && i != pending_lines.end()) {
    //     if (i->second != mf->get_inst().get_uid()) return SECTOR_MISS;
    //   }
    // }

    // 如果上面的 cache line 可以被逐出来 reserve 新访问，则返回 MISS。
    return MISS;
  }

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::process_tag_probe() 函数
  :name: code-cache-process_tag_probe

  enum cache_request_status data_cache::process_tag_probe(
      bool wr, enum cache_request_status probe_status, new_addr_type addr,
      unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events) {
    // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
    // data_cache constructor to reflect the corresponding cache configuration
    // options. Function pointers were used to avoid many long conditional
    // branches resulting from many cache configuration options.
    cache_request_status access_status = probe_status;
    if (wr) {  // Write
      if (probe_status == HIT) {
        //这里会在cache_index中写入cache block的索引。
        access_status =
            (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
      } else if ((probe_status != RESERVATION_FAIL) ||
                (probe_status == RESERVATION_FAIL &&
                  m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
        access_status =
            (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
      } else {
        // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
        // lines are reserved)
        m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
      }
    } else {  // Read
      if (probe_status == HIT) {
        access_status =
            (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
      } else if (probe_status != RESERVATION_FAIL) {
        access_status =
            (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
      } else {
        // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
        // lines are reserved)
        m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
      }
    }

    m_bandwidth_management.use_data_port(mf, access_status, events);
    return access_status;
  }


.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: tag_array::access() 函数
  :name: code-tag_array-access

  enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                              unsigned &idx, bool &wb,
                                              evicted_block_info &evicted,
                                              mem_fetch *mf) {
    // 对当前 tag_array 的访问次数加 1。
    m_access++;
    // 标记当前 tag_array 所属 cache 是否被使用过。一旦有 access() 函数被调用，则
    // 说明被使用过。
    is_used = true;
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    // 由于当前函数没有把之前 probe 函数的 cache 访问状态传参进来，这里这个 probe 
    // 单纯的重新获取这个状态。
    enum cache_request_status status = probe(addr, idx, mf, mf->is_write());
    switch (status) {
      // 新访问是 HIT_RESERVED 的话，不执行动作。
      case HIT_RESERVED:
        m_pending_hit++;
      // 新访问是 HIT 的话，设置第 idx 号 cache line 以及 mask 对应的 sector 的最
      // 末此访问时间为当前拍。
      case HIT:
        m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
        break;
      // 新访问是 MISS 的话，说明已经选定 m_lines[idx] 作为逐出并 reserve 新访问的
      // cache line。
      case MISS:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
        // For V100, L1 cache and L2 cache are all `allocate on miss`.
        // m_alloc_policy，分配策略：
        //     对于发送到 L1D cache 的请求：
        //         如果命中，则立即返回所需数据；
        //         如果未命中，则分配与缓存未命中相关的资源并将请求转至 L2 cache。
        //     allocateon-miss/fill 是两种缓存行分配策略。对于 allocateon-miss，需
        //     为未完成的未命中分配一个缓存行槽、一个 MSHR 和一个未命中队列条目。相比
        //     之下，allocate-on-fill，当未完成的未命中发生时，需要分配一个 MSHR 和
        //     一个未命中队列条目，但当所需数据从较低内存级别返回时，会选择受害者缓存
        //     行槽。在这两种策略中，如果任何所需资源不可用，则会发生预留失败，内存管
        //     道会停滞。分配的 MSHR 会被保留，直到从 L2 缓存/片外内存中获取数据，而
        //     未命中队列条目会在未命中请求转发到 L2 缓存后被释放。由于 allocate-on-
        //     fill 在驱逐之前将受害者缓存行保留在缓存中更长时间，并为未完成的未命中
        //     保留更少的资源，因此它往往能获得更多的缓存命中和更少的预留失败，从而比 
        //     allocate-on-miss 具有更好的性能。尽管填充时分配需要额外的缓冲和流控制
        //     逻辑来按顺序将数据填充到缓存中，但按顺序执行模型和写入驱逐策略使 GPU 
        //     L1D 缓存对填充时分配很友好，因为在填充时要驱逐受害者缓存时，没有脏数据
        //     写入 L2。
        //     详见 paper：
        //     The Demand for a Sound Baseline in GPU Memory Architecture Research. 
        //     https://hzhou.wordpress.ncsu.edu/files/2022/12/Hongwen_WDDD2017.pdf
        //
        //     For streaming cache: (1) we set the alloc policy to be on-fill 
        //     to remove all line_alloc_fail stalls. if the whole memory is 
        //     allocated to the L1 cache, then make the allocation to be on 
        //     MISS, otherwise, make it ON_FILL to eliminate line allocation 
        //     fails. i.e. MSHR throughput is the same, independent on the L1
        //     cache size/associativity So, we set the allocation policy per 
        //     kernel basis, see shader.cc, max_cta() function. (2) We also 
        //     set the MSHRs to be equal to max allocated cache lines. This
        //     is possible by moving TAG to be shared between cache line and 
        //     MSHR enrty (i.e. for each cache line, there is an MSHR entry 
        //     associated with it). This is the easiest think we can think of 
        //     to model (mimic) L1 streaming cache in Pascal and Volta. For 
        //     more information about streaming cache, see: 
        //     https://www2.maths.ox.ac.uk/~gilesm/cuda/lecs/VoltaAG_Oxford.pdf
        //     https://ieeexplore.ieee.org/document/8344474/
        if (m_config.m_alloc_policy == ON_MISS) {
          // 访问时遇到 MISS，说明 probe 确定的 idx 号 cache line 需要被逐出来为新
          // 访问提供 RESERVE 的空间。但是，这里需要判断 idx 号 cache line 是否是 
          // MODIFIED，如果是的话，需要执行写回，设置写回的标志为 wb = true，设置逐
          // 出 cache line 的信息。
          if (m_lines[idx]->is_modified_line()) {
            // m_lines[idx] 作为逐出并 reserve 新访问的 cache line，如果它的某个 
            // sector 已经被 MODIFIED，则需要执行写回操作，设置写回标志为 wb = true，
            // 设置逐出 cache line 的信息。
            wb = true;
            evicted.set_info(m_lines[idx]->m_block_addr,
                            m_lines[idx]->get_modified_size(),
                            m_lines[idx]->get_dirty_byte_mask(),
                            m_lines[idx]->get_dirty_sector_mask());
            // 由于执行写回操作，MODIFIED 造成的 m_dirty 数量应该减1。
            m_dirty--;
          }
          // 执行对新访问的 reserve 操作。
          m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                                time, mf->get_access_sector_mask());
        }
        break;
      // Cache block 有效，但是其中的 byte mask = Cache block[mask] 状态无效，说明
      // sector 缺失。
      case SECTOR_MISS:
        assert(m_config.m_cache_type == SECTOR);
        m_sector_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
        // For V100, L1 cache and L2 cache are all `allocate on miss`.
        if (m_config.m_alloc_policy == ON_MISS) {
          bool before = m_lines[idx]->is_modified_line();
          // 设置 m_lines[idx] 为新访问分配一个 sector。
          ((sector_cache_block *)m_lines[idx])
              ->allocate_sector(time, mf->get_access_sector_mask());
          if (before && !m_lines[idx]->is_modified_line()) {
            m_dirty--;
          }
        }
        break;
      // probe函数中：
      // all_reserved 被初始化为 true，是指所有 cache line 都没有能够逐出来为新访问
      // 提供 RESERVE 的空间，这里一旦满足函数两个 if 条件，说明 cache line 可以被逐
      // 出来提供空间供 RESERVE 新访问，这里 all_reserved 置为 false。
      // 而一旦最终 all_reserved 仍旧保持 true 的话，就说明 cache line 不可被逐出，
      // 发生 RESERVATION_FAIL。因此这里不执行任何操作。
      case RESERVATION_FAIL:
        m_res_fail++;
        shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
        break;
      default:
        fprintf(stderr,
                "tag_array::access - Error: Unknown"
                "cache_request_status %d\n",
                status);
        abort();
    }
    return status;
  }




.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::rd_miss_base() 函数
  :name: code-data_cache-rd_miss_base

  /****** Read miss functions (Set by config file) ******/

  // Baseline read miss: Send read request to lower level memory,
  // perform write-back as necessary
  /*
  READ MISS 操作。
  */
  enum cache_request_status data_cache::rd_miss_base(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status) {
    // 读 miss 时，就需要将数据请求发送至下一级存储。这里或许需要真实地向下一级存储发
    // 送读请求，也或许由于 mshr 的存在，可以将数据请求合并进去，这样就不需要真实地向
    // 下一级存储发送读请求。
    // miss_queue_full 检查是否一个 miss 请求能够在当前时钟周期内被处理，当一个请求
    // 的大小大到 m_miss_queue 放不下时即在当前拍内无法处理，发生 RESERVATION_FAIL。
    if (miss_queue_full(1)) {
      // cannot handle request this cycle (might need to generate two requests).
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      return RESERVATION_FAIL;
    }

    // m_config.block_addr(addr): 
    //     return addr & ~(new_addr_type)(m_line_sz - 1);
    // |-------|-------------|--------------|
    //            set_index   offset in-line
    // |<--------tag--------> 0 0 0 0 0 0 0 |
    new_addr_type block_addr = m_config.block_addr(addr);
    // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
    // 级存储。
    bool do_miss = false;
    // wb 代表是否需要写回（当一个被逐出的 cache block 被 MODIFIED 时，需要写回到
    // 下一级存储），evicted代表被逐出的 cache line 的信息。
    bool wb = false;
    evicted_block_info evicted;
    // READ MISS 处理函数，检查 MSHR 是否命中或者 MSHR 是否可用，依此判断是否需要
    // 向下一级存储发送读请求。
    send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                      evicted, events, false, false);
    // 如果 send_read_request 中数据请求已经被加入到 MSHR，或是原先存在该条目将请
    // 求合并进去，或是原先不存在该条目将请求插入进去，那么 do_miss 为 true，代表
    // 要将某个cache block逐出并接收 mf 从下一级存储返回的数据。
    // m_lines[idx] 作为逐出并 reserve 新访问的 cache line，如果它的某个 sector 
    // 已经被MODIFIED，则需要执行写回操作，设置写回的标志为 wb = true。
    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level).
      // 这里如果 cache 的写策略为写直达，就不需要在读 miss 时将被逐出的 MODIFIED 
      // cache block 写回到下一级存储，因为这个 cache block 在被 MODIFIED 的时候
      // 已经被 write-through 到下一级存储了。
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        // 发送写请求，将 MODIFIED 的被逐出的 cache block 写回到下一级存储。
        // 在 V100 中，
        //     m_wrbk_type：L1 cache 为 L1_WRBK_ACC，L2 cache 为 L2_WRBK_ACC。
        //     m_write_policy：L1 cache 为 WRITE_THROUGH。
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        // the evicted block may have wrong chip id when advanced L2 hashing 
        // is used, so set the right chip address from the original mf.
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        // 将数据写请求一同发送至下一级存储。
        // 需要做的是将读请求类型 WRITE_BACK_REQUEST_SENT放 入events，并将数据请
        // 求 mf 放入当前 cache 的 m_miss_queue 中，等 baseline_cache::cycle() 
        // 将队首的数据请求 mf 发送给下一级存储。
        send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: baseline_cache::send_read_request() 函数
  :name: code-baseline_cache-send_read_request

  // Read miss handler. Check MSHR hit or MSHR available
  /*
  READ MISS 处理函数，检查 MSHR 是否命中或者 MSHR 是否可用，依此判断是否需要向下一
  级存储发送读请求。
  */
  void baseline_cache::send_read_request(new_addr_type addr,
                                         new_addr_type block_addr,
                                         unsigned cache_index, mem_fetch *mf,
                                         unsigned time, bool &do_miss, bool &wb,
                                         evicted_block_info &evicted,
                                         std::list<cache_event> &events,
                                         bool read_only, bool wa) {
    // 1. 如果是 Sector Cache：
    //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
    //  位 + sector offset 位。即除 single sector byte offset 位 以外的所有位。
    //  |<----------mshr_addr----------->|
    //                     sector offset  off in-sector
    //                     |-------------|-----------|
    //                      \                       /
    //                       \                     /
    //  |-------|-------------|-------------------|
    //             set_index     offset in-line
    //  |<----tag----> 0 0 0 0|
    // 2. 如果是 Line Cache：
    //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
    //  位。即除 single line byte off-set 位 以外的所有位。
    //  |<----mshr_addr--->|
    //                              line offset
    //                     |-------------------------|
    //                      \                       /
    //                       \                     /
    //  |-------|-------------|-------------------|
    //             set_index     offset in-line
    //  |<----tag----> 0 0 0 0|
    //
    // mshr_addr 定义：
    //   new_addr_type mshr_addr(new_addr_type addr) const {
    //     return addr & ~(new_addr_type)(m_atom_sz - 1);
    //   }
    // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
    // 其中 SECTOR_SIZE = const (32 bytes per sector).
    new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
    // 这里实际上是 MSHR 查找是否已经有 mshr_addr 的请求被合并到 MSHR。如果已经被挂
    // 起则 mshr_hit = true。需要注意，MSHR 中的条目是以 mshr_addr 为索引的，即来自
    // 同一个 line（对于非 Sector Cache）或者来自同一个 sector（对于 Sector Cache）
    // 的事务被合并，因为这种 cache 所请求的最小单位分别是一个 line 或者一个 sector，
    // 因此没必要发送那么多事务，只需要发送一次即可。
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    // 如果 mshr_addr 在 MSHR 中已存在条目，m_mshrs.full 检查是否该条目的合并数量已
    // 达到最大合并数；如果 mshr_addr 在 MSHR 中不存在条目，则检查是否有空闲的 MSHR 
    // 条目可以将 mshr_addr 插入进 MSHR。
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (mshr_hit && mshr_avail) {
      // 如果 MSHR 命中，且 mshr_addr 对应条目的合并数量没有达到最大合并数，则将数据
      // 请求 mf 加入到 MSHR 中。
      if (read_only)
        m_tag_array->access(block_addr, time, cache_index, mf);
      else
        // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

      // 将 mshr_addr 地址的数据请求 mf 加入到 MSHR 中。因为命中 MSHR，说明前面已经
      // 有对该数据的请求发送到下一级缓存了，因此这里只需要等待前面的请求返回即可。
      m_mshrs.add(mshr_addr, mf);
      m_stats.inc_stats(mf->get_access_type(), MSHR_HIT);
      // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
      // 级存储。
      do_miss = true;

    } else if (!mshr_hit && mshr_avail &&
              (m_miss_queue.size() < m_config.m_miss_queue_size)) {
      // 如果 MSHR 未命中，但有空闲的 MSHR 条目可以将 mshr_addr 插入进 MSHR，则将数
      // 据请求 mf 插入到 MSHR 中。
      // 对于 L1 cache 和 L2 cache，read_only 为 false，对于 read_only_cache 来说，
      // read_only 为true。
      if (read_only)
        m_tag_array->access(block_addr, time, cache_index, mf);
      else
        // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

      // 将 mshr_addr 地址的数据请求 mf 加入到 MSHR 中。因为没有命中 MSHR，因此还需
      // 要将该数据的请求发送到下一级缓存。
      m_mshrs.add(mshr_addr, mf);
      // if (m_config.is_streaming() && m_config.m_cache_type == SECTOR) {
      //   m_tag_array->add_pending_line(mf);
      // }
      // 设置 m_extra_mf_fields[mf]，意味着如果 mf 在 m_extra_mf_fields 中存在，即 
      // mf 等待着下一级存储的数据回到当前缓存填充。
      m_extra_mf_fields[mf] = extra_mf_fields(
          mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
      mf->set_data_size(m_config.get_atom_sz());
      mf->set_addr(mshr_addr);
      // mf 为 miss 的请求，加入 miss_queue，MISS 请求队列。
      // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递给下
      // 一层存储。因为没有命中 MSHR，说明前面没有对该数据的请求发送到下一级缓存，
      // 因此这里需要把该请求发送给下一级存储。
      m_miss_queue.push_back(mf);
      mf->set_status(m_miss_queue_status, time);
      // 在 V100 配置中，wa 对 L1/L2/read_only cache 均为 false。
      if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));
      // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
      // 级存储。
      do_miss = true;
    } else if (mshr_hit && !mshr_avail)
      // 如果 MSHR 命中，但 mshr_addr 对应条目的合并数量达到了最大合并数。
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
    else if (!mshr_hit && !mshr_avail)
      // 如果 MSHR 未命中，且 mshr_addr 没有空闲的 MSHR 条目可将 mshr_addr 插入。
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
    else
      assert(0);
  }

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::send_write_request() 函数
  :name: code-data_cache-send_write_request

  // Sends write request to lower level memory (write or writeback)
  /*
  将数据写请求一同发送至下一级存储。这里需要做的是将写请求类型 WRITE_REQUEST_SENT 或 
  WRITE_BACK_REQUEST_SENT 放入 events，并将数据请求 mf 放入 m_miss_queue中，等待下
  一时钟周期 baseline_cache::cycle() 将队首的数据请求 mf 发送给下一级存储。
  */
  void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                      unsigned time,
                                      std::list<cache_event> &events) {
    events.push_back(request);
    // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递给下
    // 一级存储。
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
  }

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::wr_hit_wb() 函数
  :name: code-data_cache-wr_hit_wb

  /****** Write-hit functions (Set by config file) ******/

  // Write-back hit: Mark block as modified
  /*
  若 Write Hit 时采取 write-back 策略，则需要将数据单写入 cache，不需要直接将数据写入
  下一级存储。等到新数据 fill 进来时，再将旧数据逐出并写入下一级存储。
  */
  cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                             unsigned cache_index, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events,
                                             enum cache_request_status status) {
    // m_config.block_addr(addr): 
    //     return addr & ~(new_addr_type)(m_line_sz - 1);
    // |-------|-------------|--------------|
    //            set_index   offset in-line
    // |<--------tag--------> 0 0 0 0 0 0 0 |
    // write-back 策略不需要直接将数据写入下一级存储，因此不需要调用miss_queue_full()
    // 以及 send_write_request() 函数来发送写回请求到下一级存储。
    new_addr_type block_addr = m_config.block_addr(addr);
    // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
    m_tag_array->access(block_addr, time, cache_index, mf);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    // 如果 block 不是 modified line，则增加 dirty 计数。因为如果这个时候 block 不是
    // modified line，说明这个 block 是 clean line，而现在要写入数据，因此需要将这个
    // block 设置为 modified line。这样的话，dirty 计数就需要增加。但如果 block 已经
    // 是 modified line，则不需要增加 dirty 计数，因为这个 block 在上次变成 dirty 的
    // 时候，dirty 计数已经增加过了。
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // 设置 block 的状态为 modified，即将 block 设置为 MODIFIED。这样的话，下次再有
    // 数据请求访问这个 block 的时候，就可以直接从 cache 中读取数据，而不需要再次访问
    // 下一级存储。当然，当有下次填充进这个 block 的数据请求时（block 的 tag 与请求的
    // tag 不一致），由于这个 block 的状态已经被设置为 modified，因此需要将此 block 
    // 的数据逐出并写回到下一级存储。
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    // 更新一个 cache block 的状态为可读。但需要注意的是，这里的可读是指该 sector 可
    // 读，而不是整个 block 可读。如果一个 sector 内的所有的 byte mask 位全都设置为 
    // dirty 了，则将该sector 可设置为可读，因为当前的 sector 已经是全部更新为最新值
    // 了，是可读的。这个函数对所有的数据请求 mf 的所有访问的 sector 进行遍历，这里的
    // sector 是由 mf 访问的，并由 mf->get_access_sector_mask() 确定。
    update_m_readable(mf,cache_index);

    return HIT;
  }

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::wr_hit_wt() 函数
  :name: code-data_cache-wr_hit_wt

  // Write-through hit: Directly send request to lower level memory
  /*
  若 Write Hit 时采取 write-through 策略的话，则需要将数据不单单写入 cache，还需要直
  接将数据写入下一级存储。
  */
  cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                             unsigned cache_index, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events,
                                             enum cache_request_status status) {
    // miss_queue_full 检查是否一个 miss 请求能够在当前时钟周期内被处理，当一个请求的
    // 大小大到 m_miss_queue 放不下时即在当前拍内无法处理，发生 RESERVATION_FAIL。
    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      // 如果 miss_queue 满了，但由于 write-through 策略要求数据应该直接写入下一级存
      // 储，因此这里返回 RESERVATION_FAIL，表示当前时钟周期内无法处理该请求。
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }
    // m_config.block_addr(addr): 
    //     return addr & ~(new_addr_type)(m_line_sz - 1);
    // |-------|-------------|--------------|
    //            set_index   offset in-line
    // |<--------tag--------> 0 0 0 0 0 0 0 |
    new_addr_type block_addr = m_config.block_addr(addr);
    // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
    m_tag_array->access(block_addr, time, cache_index, mf);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    // 如果 block 不是 modified line，则增加 dirty 计数。因为如果这个时候 block 不是
    // modified line，说明这个 block 是 clean line，而现在要写入数据，因此需要将这个
    // block 设置为 modified line。这样的话，dirty 计数就需要增加。但如果 block 已经
    // 是 modified line，则不需要增加 dirty 计数，因为这个 block 在上次变成 dirty 的
    // 时候，dirty 计数已经增加过了。
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // 设置 block 的状态为 modified，即将 block 设置为 MODIFIED。这样的话，下次再有
    // 数据请求访问这个 block 的时候，就可以直接从 cache 中读取数据，而不需要再次访问
    // 下一级存储。
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    // 更新一个 cache block 的状态为可读。但需要注意的是，这里的可读是指该 sector 可
    // 读，而不是整个 block 可读。如果一个 sector 内的所有的 byte mask 位全都设置为 
    // dirty 了，则将该sector 可设置为可读，因为当前的 sector 已经是全部更新为最新值
    // 了，是可读的。这个函数对所有的数据请求 mf 的所有访问的 sector 进行遍历，这里的
    // sector 是由 mf 访问的，并由 mf->get_access_sector_mask() 确定。
    update_m_readable(mf,cache_index);

    // generate a write-through
    // write-through 策略需要将数据写入 cache 的同时也直接写入下一级存储。这里需要做
    // 的是将写请求类型 WRITE_REQUEST_SENT 放入 events，并将数据请求放入当前 cache  
    // 的 m_miss_queue 中，等待baseline_cache::cycle() 将 m_miss_queue 队首的数
    // 据写请求 mf 发送给下一级存储。
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

    return HIT;
  }


.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::wr_hit_we() 函数
  :name: code-data_cache-wr_hit_we

  // Write-evict hit: Send request to lower level memory and invalidate
  // corresponding block
  /*
  写逐出命中：向下一级存储发送写回请求并直接逐出相应的 cache block 并设置其无效。
  */
  cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                             unsigned cache_index, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events,
                                             enum cache_request_status status) {
    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    // generate a write-through/evict
    cache_block_t *block = m_tag_array->get_block(cache_index);
    // write-evict 策略需要将 cache block 直接逐出置为无效的同时也直接写入下一级存
    // 储。这里需要做的是将写请求类型 WRITE_REQUEST_SENT 放入 events，并将数据请求  
    // 放入 m_miss_queue 中，等待baseline_cache::cycle() 将 m_miss_queue 队首的
    // 数据写请求 mf 发送给下一级存储。
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

    // Invalidate block
    // 写逐出，将 cache block 直接逐出置为无效。
    block->set_status(INVALID, mf->get_access_sector_mask());

    return HIT;
  }


.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::wr_hit_global_we_local_wb() 函数
  :name: code-data_cache-wr_hit_global_we_local_wb

  // Global write-evict, local write-back: Useful for private caches
  /*
  全局访存采用写逐出，本地访存采用写回。这种策略适用于私有缓存。这个策略比较简单，即只
  需要判断当前的数据请求是全局访存还是本地访存，然后分别采用写逐出和写回策略即可。
  */
  enum cache_request_status data_cache::wr_hit_global_we_local_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status) {
    bool evict = (mf->get_access_type() ==
                  GLOBAL_ACC_W); // evict a line that hits on global memory write
    if (evict)
      return wr_hit_we(addr, cache_index, mf, time, events,
                      status); // Write-evict
    else
      return wr_hit_wb(addr, cache_index, mf, time, events,
                      status); // Write-back
  }



.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::wr_miss_wa_naive() 函数
  :name: code-data_cache-wr_miss_wa_naive

  /****** Write-miss functions (Set by config file) ******/

  // Write-allocate miss: Send write request to lower level memory
  // and send a read request for the same block
  /*
  GPGPU-Sim 3.x版本中的naive写分配策略。wr_miss_wa_naive 策略在写 MISS 时，需要先将 
  mf 数据包直接写入下一级存储，即它会将 WRITE_REQUEST_SENT 放入 events，并将数据请求 
  mf 放入 m_miss_queue 中，等待下一个周期 baseline_cache::cycle() 将 m_miss_queue 
  队首的数据包 mf 发送给下一级存储。其次，wr_miss_wa_naive 策略还会将 addr 地址的数据
  读到当前 cache 中，这时候会执行 send_read_request 函数。但是在 send_read_request 
  函数中，很有可能这个读请求需要 evict 一个 block 才可以将新的数据读入到 cache 中，这
  时候如果 evicted block 是 modified line，则需要将这个 evicted block 写回到下一级
  存储，这时候会根据 do_miss 和 wb 的值执行 send_write_request 函数。
  */
  enum cache_request_status data_cache::wr_miss_wa_naive(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status) {
    // m_config.block_addr(addr): 
    //     return addr & ~(new_addr_type)(m_line_sz - 1);
    // |-------|-------------|--------------|
    //            set_index   offset in-line
    // |<--------tag--------> 0 0 0 0 0 0 0 | 
    new_addr_type block_addr = m_config.block_addr(addr);
    // 1. 如果是 Sector Cache：
    //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
    //  位 + sector offset 位。即除 single sector byte offset 位 以外的所有位。
    //  |<----------mshr_addr----------->|
    //                     sector offset  off in-sector
    //                     |-------------|-----------|
    //                      \                       /
    //                       \                     /
    //  |-------|-------------|-------------------|
    //             set_index     offset in-line
    //  |<----tag----> 0 0 0 0|
    // 2. 如果是 Line Cache：
    //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
    //  位。即除 single line byte off-set 位 以外的所有位。
    //  |<----mshr_addr--->|
    //                              line offset
    //                     |-------------------------|
    //                      \                       /
    //                       \                     /
    //  |-------|-------------|-------------------|
    //             set_index     offset in-line
    //  |<----tag----> 0 0 0 0|
    //
    // mshr_addr 定义：
    //   new_addr_type mshr_addr(new_addr_type addr) const {
    //     return addr & ~(new_addr_type)(m_atom_sz - 1);
    //   }
    // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
    // 其中 SECTOR_SIZE = const (32 bytes per sector).
    new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

    // Write allocate, maximum 3 requests (write miss, read request, write back
    // request) Conservatively ensure the worst-case request can be handled this
    // cycle.
    // MSHR 的 m_data 的 key 中存储了各个合并的地址，probe() 函数主要检查是否命中，
    // 即主要检查 m_data.keys() 这里面有没有 mshr_addr。
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    // 首先查找是否 MSHR 表中有 block_addr 地址的条目。如果存在该条目（命中 MSHR），
    // 看是否有空间合并进该条目。如果不存在该条目（未命中 MSHR），看是否有其他空间允
    // 许添加 mshr_addr 这一条目。
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递给下一
    // 级存储。因此当遇到 miss 的请求或者写回的请求需要访问下一级存储时，会把 miss 的
    // 请求放到 m_miss_queue 中。
    //   bool miss_queue_full(unsigned num_miss) {
    //     return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
    //   }
    
    // 如果 m_miss_queue.size() 已经不能容下三个数据包的话，有可能无法完成后续动作，
    // 因为后面最多需要执行三次 send_write_request，在 send_write_request 里每执行
    // 一次，都需要向 m_miss_queue 添加一个数据包。
    // Write allocate, maximum 3 requests (write miss, read request, write back
    // request) Conservatively ensure the worst-case request can be handled this
    // cycle.
    if (miss_queue_full(2) || 
        // 如果 miss_queue_full(2) 返回 false，有空余空间支持执行三次 send_write_
        // request，那么就需要看 MSHR 是否有可用空间。后面这串判断条件其实可以化简成 
        // if (miss_queue_full(2) || !mshr_avail)。
        // 即符合 RESERVATION_FAIL 的条件：
        //   1. m_miss_queue 不足以放入三个 WRITE_REQUEST_SENT 请求；
        //   2. MSHR 不能合并请求（未命中，或者没有可用空间添加新条目）。
        (!(mshr_hit && mshr_avail) &&
        !(!mshr_hit && mshr_avail &&
          (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(2))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
      else
        assert(0);

      // 符合 RESERVATION_FAIL 的条件：
      //   1. m_miss_queue 不足以放入三个 WRITE_REQUEST_SENT 请求；
      //   2. MSHR 不能合并请求（未命中，或者没有可用空间添加新条目）。
      return RESERVATION_FAIL;
    }

    // send_write_request 执行：
    //   events.push_back(request);
    //   // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递
    //   // 给下一级存储。
    //   m_miss_queue.push_back(mf);
    //   mf->set_status(m_miss_queue_status, time);
    // wr_miss_wa_naive 策略在写 MISS 时，需要先将 mf 数据包直接写入下一级存储，即它
    // 会将 WRITE_REQUEST_SENT 放入 events，并将数据请求 mf 放入 m_miss_queue 中，
    // 等待下一个周期 baseline_cache::cycle() 将 m_miss_queue 队首的数据包 mf 发送
    // 给下一级存储。其次，wr_miss_wa_naive 策略还会将 addr 地址的数据读到当前 cache
    // 中，这时候会执行 send_read_request 函数。但是在 send_read_request 函数中，很
    // 有可能这个读请求需要 evict 一个 block 才可以将新的数据读入到 cache 中，这时候
    // 如果 evicted block 是 modified line，则需要将这个 evicted block 写回到下一级
    // 存储，这时候会根据 do_miss 和 wb 的值执行 send_write_request 函数。
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
    // Tries to send write allocate request, returns true on success and false on
    // failure
    // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
    //    return RESERVATION_FAIL;

    const mem_access_t *ma =
        new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                        false,  // Now performing a read
                        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf =
        new mem_fetch(*ma, NULL, mf->get_ctrl_size(), mf->get_wid(),
                      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
                      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

    // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
    // 级存储。
    bool do_miss = false;
    // wb 变量标识 tag_array::access() 函数中，如果下面的 send_read_request 函数
    // 发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级存储。
    // 如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配给新访问
    // 之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这个 block 
    // 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一级存储。这个 
    // evicted block 的信息被设置在 evicted 中。
    bool wb = false;
    evicted_block_info evicted;

    // Send read request resulting from write miss
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    // do_miss 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期
    // 发送到下一级存储。
    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      // wb 变量标识 tag_array::access() 函数中，如果下面的 send_read_request 函
      // 数发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级存
      // 储。如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配给
      // 新访问之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这个  
      // block 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一级存 
      // 储。这个 evicted block 的信息被设置在 evicted 中。
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        assert(status ==
              MISS); // SECTOR_MISS and HIT_RESERVED should not send write back
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        // 将 tag_array::access() 函数中逐出的 evicted block 写回到下一级存储。
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                          time, events);
      }
      // 如果 do_miss 为 true，表示请求被填充进 MSHR 或者 被放到 m_miss_queue 以在
      // 下一个周期发送到下一级存储。即整个写 MISS 处理函数的所有过程全部完成，返回的
      // 是 write miss 这个原始写请求的状态。
      return MISS;
    }

    // 如果 do_miss 为 false，表示请求未被填充进 MSHR 或者 未被放到 m_miss_queue 以
    // 在下一个周期发送到下一级存储。即整个写 MISS 处理函数没有将读请求发送出去，因此
    // 返回 RESERVATION_FAIL。
    return RESERVATION_FAIL;
  }

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: data_cache::wr_miss_no_wa() 函数
  :name: code-data_cache-wr_miss_no_wa

  // No write-allocate miss: Simply send write request to lower level memory
  /*
  No write-allocate miss，这个处理函数仅仅简单地将写请求发送到下一级存储。
  */
  enum cache_request_status data_cache::wr_miss_no_wa(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status) {
    // 如果 m_miss_queue.size() 已经不能容下一个数据包的话，有可能无法完成后续动作，
    // 因为后面最多需要执行一次 send_write_request，在 send_write_request 里每执行
    // 一次，都需要向 m_miss_queue 添加一个数据包。
    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    // on miss, generate write through (no write buffering -- too many threads 
    // for that)
    // send_write_request 执行：
    //   events.push_back(request);
    //   // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递
    //   // 给下一级存储。
    //   m_miss_queue.push_back(mf);
    //   mf->set_status(m_miss_queue_status, time);
    // No write-allocate miss 策略在写 MISS 时，直接将 mf 数据包直接写入下一级存储。
    // 这里需要做的是将写请求类型 WRITE_REQUEST_SENT 放入 events，并将数据请求放入  
    // m_miss_queue 中，等待baseline_cache::cycle() 将 m_miss_queue 队首的数据写
    // 请求 mf 发送给下一级存储。
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

    return MISS;
  }

  
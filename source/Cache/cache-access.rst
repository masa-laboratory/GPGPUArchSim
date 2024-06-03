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
  // m_index_function：set index 的计算函数，GV100 架构中的 L1D Cache 的配置为 LINEAR_SET_FUNCTION。
  unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                       unsigned m_line_sz_log2,
                                       unsigned m_nset_log2,
                                       unsigned m_index_function) const {
    unsigned set_index = 0;
    switch (m_index_function) {
      // ......
      case LINEAR_SET_FUNCTION: {
        // addr: [m_line_sz_log2-1:0]                            => byte offset
        // addr: [m_line_sz_log2+m_nset_log2-1:m_line_sz_log2]   => set index
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
    // 保留成功，当访问地址被映射到一个已经被分配的 cache block/sector 时，cache block/sector 的
    // 状态被设置为 RESERVED。
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
    // |<--------tag-------->|
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
.. _Cache-Access:

Cache的访问
============

Cache作为一种高速存储资源，通过存储近期或频繁访问的数据，极大地减少了处理器与主存之间的数据传输时间，从而提升了系统的整体性能。本章主要介绍 GPGPU-Sim 如何模拟对 Cache 的访问过程。

.. _Cache-Basic-Knowledge:

Cache Block基础知识
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
+++++++++++++

本节主要介绍一个访存地址如何映射到一个 cache block。如何将一个特定的访存地址映射到这些 cache block 上，是通过一系列精确计算的步骤完成的。其中最关键的两步是计算 `Tag 位`` 和 `Set Index 位`。 `Tag 位` 用于标识数据存储在哪个缓存块中，而 `Set Index 位` 则用于确定这个地址映射到缓存中的哪一个 Set 上。不过，即使能够确定一个地址映射到了哪个 Set，要找到这个数据是否已经被缓存（以及具体存储在哪一路），还需进行遍历查找，这一步是判断缓存命中还是缓存失效的关键环节。接下来，将详细介绍 `Tag 位` 和 `Set Index 位` 的计算过程，这两者是理解地址映射机制的重点。

Tag 位的计算
^^^^^^^^^^^^

下面的代码块 :ref:`fig-tag-calc` 展示了如何由访存地址 `addr` 计算 `tag 位`。

这里需要注意的是，最新版本中的 GPGPU-Sim 中的 `tag 位` 是由 `index 位` 和 `traditional tag 位` 共同组成的（这里所说的 `traditional tag 位` 就是指传统 CPU 上 Cache 的 `tag 位` 的计算方式： ``traditional tag = addr >> (log2(m_line_sz) + log2(m_nset))``），其中 `m_line_sz` 和 `m_nset` 分别是 Cache 的 `line size` 和 `set` 的数量），这样可以允许更复杂的 `set index 位` 的计算，从而避免将 `set index 位` 不同但是 `traditional tag 位` 相同的地址映射到同一个 `set`。这里是把完整的 [`traditional tag 位 + set index 位 + log2(m_line_sz)'b0`] 来作为 `tag 位`。

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

Set Index 位的计算
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^

下面的代码块 :ref:`fig-cache-block-addrmap` 用一个访存地址 `addr` 展示了访问 Cache 的地址映射。

.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: Cache Block的地址映射
  :name: fig-cache-block-addrmap
  
  // 1. 如果是 Line Cache：
  //  MSHR 的地址即为地址 addr 的 [tag 位 + set_index 位]。即除 offset in-line 位以外的所有位。
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //  |<-Tr-->|            \                     /
  //  |-------|-------------|-------------------| // addr
  //             set_index     offset in-line
  //  |<--------tag--------> 0 0 0 0 0 0 0 0 0 0|
  // 2. 如果是 Sector Cache：
  //  MSHR 的地址即为地址 addr 的 [tag 位 + set_index 位 + sector offset 位]。即除 offset in-
  //  sector 位以外的所有位。
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
+++++++++++++++++
















在访问Cache的时候，会调用 ``access()`` 函数，例如 ``m_L2cache->access()``，``m_L1I->access()``，``m_L1D->access()`` 等。

然后 Cache 会调用 ``tag_array::probe()`` 函数来判断 Cache 的访问状态，返回的状态有以下几种：

- **HIT_RESERVED** ：对于Sector Cache来说，如果Cache block[mask]状态是RESERVED，说明有其他的线程正在读取这个Cache block。挂起的命中访问已命中处于RESERVED状态的缓存行，这意味着同一行上已存在由先前缓存未命中发送的flying内存请求。
- **HIT** ：
- **SECTOR_MISS** ：
- **RESERVATION_FAIL** ：
- **MISS** ：
- **MSHR_HIT** ：


.. code-block:: c
  :lineno-start: 0
  :emphasize-lines: 0
  :linenos:
  :caption: shader.cc
  :name: c-code 

  enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                             mem_access_sector_mask_t mask,
                                             bool is_write, bool probe_mode,
                                             mem_fetch *mf) const {
    //这里的输入地址addr是cache block的地址，该地址即为地址addr的tag位+set index位。即除
    //offset位以外的所有位。
    //  |-------|-------------|--------------|
    //     tag     set_index   offset in-line

    // assert( m_config.m_write_policy == READ_ONLY );
    //返回一个地址addr在Cache中的set index。这里的set index有一整套的映射函数。
    unsigned set_index = m_config.set_index(addr);
    //为了便于起见，这里的标记包括index和Tag。这允许更复杂的（可能导致不同的indexes映射到
    //同一set）set index计算，因此需要完整的标签 + 索引来检查命中/未命中。Tag现在与块地址
    //相同。
    //这里实际返回的是{除offset位以外的所有位, offset'b0}，即set index也作为tag的一部分了。
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned long long valid_timestamp = (unsigned)-1;

    bool all_reserved = true;

    // check for hit or pending hit
    //对所有的Cache Ways检查。需要注意这里其实是针对一个set的所有way进行检查，因为给我们一个
    //地址，我们可以确定它所在的set index，然后再通过tag来确定这个地址在哪一个way上。
    for (unsigned way = 0; way < m_config.m_assoc; way++) {
      // For example, 4 sets, 6 ways:
      // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
      // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
      // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
      // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
      //                |--------> index => cache_block_t *line
      // cache block的索引。
      unsigned index = set_index * m_config.m_assoc + way;
      cache_block_t *line = m_lines[index];
      // Tag相符。m_tag和tag均是：{除offset位以外的所有位, offset'b0}
      if (line->m_tag == tag) {
        // enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };
        // cache block的状态，包含：
        //   INVALID: Cache block有效，但是其中的byte mask=Cache block[mask]状态INVALID，
        //           说明sector缺失。
        //   MODIFIED: 如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当
        //             前访问也是写操作的话即为命中，但如果不是写操作则需要判断是否mask标志的
        //             块是否修改完毕，修改完毕则为命中，修改不完成则为SECTOR_MISS。因为L1 
        //             cache与L2 cache写命中时，采用write-back策略，只将数据写入该block，
        //             并不直接更新下级存储，只有当这个块被替换时，才将数据写回下级存储。
        //   VALID: 如果Cache block[mask]状态是VALID，说明已经命中。
        //   RESERVED: 为尚未完成的缓存未命中的数据提供空间。Cache block[mask]状态RESERVED，
        //             说明有其他的线程正在读取这个Cache block。挂起的命中访问已命中处于RE-
        //             SERVED状态的缓存行，意味着同一行上已存在由先前缓存未命中发送的flying
        //             内存请求。
        if (line->get_status(mask) == RESERVED) {
          //如果Cache block[mask]状态是RESERVED，说明有其他的线程正在读取这个Cache block。
          //挂起的命中访问已命中处于RESERVED状态的缓存行，这意味着同一行上已存在由先前缓存未
          //命中发送的flying内存请求。
          idx = index;
          return HIT_RESERVED;
        } else if (line->get_status(mask) == VALID) {
          //如果Cache block[mask]状态是VALID，说明已经命中。
          idx = index;
          return HIT;
        } else if (line->get_status(mask) == MODIFIED) {
          //如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当前访问也是写
          //操作的话即为命中，但如果不是写操作则需要判断是否mask标志的块是否修改完毕，修改完毕
          //则为命中，修改不完成则为SECTOR_MISS。因为L1 cache与L2 cache写命中时，采用write-
          //back策略，只将数据写入该block，并不直接更新下级存储，只有当这个块被替换时，才将数
          //据写回下级存储。
          //is_readable(mask)是判断mask标志的sector是否已经全部写完成，因为在修改cache的过程
          //中，有一个sector被修改即算作当前cache块MODIFIED，但是修改过程可能不是一下就能写完，
          //因此需要判断一下是否全部当前mask标记所读的sector写完才可以算作读命中。
          if ((!is_write && line->is_readable(mask)) || is_write) {
            // 当前line的mask位被修改，如果是写就无所谓，它依然命中，直接覆盖写即可；但是如果
            // 是读，就需要看mask位是否是可读的。如果是可读的，即为命中。
            idx = index;
            return HIT;
          } else {
            // 满足这个分支的条件是：is_write为false，当前访问是读，line->is_readable(mask)
            // 为false，mask位不是可读的，则说明当前读的sector缺失。
            idx = index;
            return SECTOR_MISS;
          }
        } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
          // 对于line cache不会走这个分支，因为line cache中，line->is_valid_line()返回的是
          // m_status的值，当其为 VALID 时，line cache中line->get_status(mask)也是返回的
          // 也是m_status的值，即为 VALID，因此对于line cache这条分支无效。
          // 但是对于sector cache， 有：
          //   virtual bool is_valid_line() { return !(is_invalid_line()); }
          // 而sector cache中的is_invalid_line()是，只要有一个sector不为INVALID即返回false，
          // 因此is_valid_line()返回的是，只要有一个sector不为INVALID就设置is_valid_line()
          // 为真。所以这条分支对于sector cache是可走的。
          //Cache block有效，但是其中的byte mask=Cache block[mask]状态无效，说明sector缺失。
          idx = index;
          return SECTOR_MISS;
        } else {
          assert(line->get_status(mask) == INVALID);
        }
      }
      
      //每一次循环中能走到这里的，即为当前cache block的line->m_tag!=tag。那么就需要考虑当前这
      //cache block能否被逐出替换，请注意，这个判断是在对每一个way循环的过程中进行的，也就是说，
      //加入第一个cache block没有返回以上访问状态，但有可能直到所有way的最后一个cache block才
      //满足line->m_tag!=tag，但是在对第0~way-2号的cache block循环判断的时候，就需要记录下每
      //一个way的cache block是否能够被逐出。因为如果等到所有way的cache block都没有满足line->
      //m_tag!=tag时，再回过头来循环所有way找最优先被逐出的cache block那就增加了模拟的开销。
      //因此实际上对于所有way中的每一个cache block，只要它不满足line->m_tag!=tag，就在这里判
      //断它能否被逐出。
      // cache block的状态，包含：
      //   INVALID: Cache block有效，但是其中的byte mask=Cache block[mask]状态INVALID，
      //           说明sector缺失。
      //   MODIFIED: 如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当
      //             前访问也是写操作的话即为命中，但如果不是写操作则需要判断是否mask标志的
      //             块是否修改完毕，修改完毕则为命中，修改不完成则为SECTOR_MISS。因为L1 
      //             cache与L2 cache写命中时，采用write-back策略，只将数据写入该block，
      //             并不直接更新下级存储，只有当这个块被替换时，才将数据写回下级存储。
      //   VALID: 如果Cache block[mask]状态是VALID，说明已经命中。
      //   RESERVED: 为尚未完成的缓存未命中的数据提供空间。Cache block[mask]状态RESERVED，
      //             说明有其他的线程正在读取这个Cache block。挂起的命中访问已命中处于RE-
      //             SERVED状态的缓存行，意味着同一行上已存在由先前缓存未命中发送的flying
      //             内存请求。
      //line->is_reserved_line()：只要有一个sector是RESERVED，就认为这个Cache Line是RESERVED。
      //这里即整个line没有sector是RESERVED。
      if (!line->is_reserved_line()) {
        // percentage of dirty lines in the cache
        // number of dirty lines / total lines in the cache
        float dirty_line_percentage =
            ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
        // If the cacheline is from a load op (not modified), 
        // or the total dirty cacheline is above a specific value,
        // Then this cacheline is eligible to be considered for replacement candidate
        // i.e. Only evict clean cachelines until total dirty cachelines reach the limit.
        //m_config.m_wr_percent在V100中配置为25%。
        //line->is_modified_line()：只要有一个sector是MODIFIED，就认为这个cache line是MODIFIED。
        //这里即整个line没有sector是MODIFIED，或者dirty_line_percentage超过m_wr_percent。
        if (!line->is_modified_line() ||
            dirty_line_percentage >= m_config.m_wr_percent) 
        {
          //一个cache line的状态有：INVALID = 0, RESERVED, VALID, MODIFIED，如果它是VALID，
          //就在上面的代码命中了。
          //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
          //被MODIFIED，来逐出；但是如果dirty的cache line的比例超过m_wr_percent（V100中配置为
          //25%），也可以不满足MODIFIED的条件。
          //在缓存管理机制中，优先逐出未被修改（"干净"）的缓存块的策略，是基于几个重要的考虑：
          // 1. 减少写回成本：缓存中的数据通常来源于更低速的后端存储（如主存储器）。当缓存块被修改
          //   （即包含"脏"数据）时，在逐出这些块之前，需要将这些更改写回到后端存储以确保数据一致性。
          //    相比之下，未被修改（"干净"）的缓存块可以直接被逐出，因为它们的内容已经与后端存储一
          //    致，无需进行写回操作。这样就避免了写回操作带来的时间和能量开销。
          // 2. 提高效率：写回操作相对于读取操作来说，是一个成本较高的过程，不仅涉及更多的时间延迟，
          //    还可能占用宝贵的带宽，影响系统的整体性能。通过先逐出那些"干净"的块，系统能够在维持
          //    数据一致性的前提下，减少对后端存储带宽的需求和写回操作的开销。
          // 3. 优化性能：选择逐出"干净"的缓存块还有助于维护缓存的高命中率。理想情况下，缓存应当存
          //    储访问频率高且最近被访问的数据。逐出"脏"数据意味着这些数据需要被写回，这个过程不仅
          //    耗时而且可能导致缓存暂时无法服务其他请求，从而降低缓存效率。
          // 4. 数据安全与完整性：在某些情况下，"脏"缓存块可能表示正在进行的写操作或者重要的数据更
          //    新。通过优先逐出"干净"的缓存块，可以降低因为缓存逐出导致的数据丢失或者完整性破坏的
          //    风险。
          
          //all_reserved被初始化为true，是指所有cache line都没有能够逐出来为新访问提供RESERVE
          //的空间，这里一旦满足上面两个if条件，说明当前line可以被逐出来提供空间供RESERVE新访问，
          //这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明当前set里
          //没有哪一个way的cache block可以被逐出，发生RESERVATION_FAIL。
          all_reserved = false;
          //line->is_invalid_line()是所有sector都无效。
          if (line->is_invalid_line()) {
            //当然了，尽管我们有LRU或者FIFO替换策略，但是最理想的情况还是优先替换整个cache block
            //都无效的块。因为这种无效的块不需要写回，能够节省带宽。
            invalid_line = index;
          } else {
            // valid line : keep track of most appropriate replacement candidate
            if (m_config.m_replacement_policy == LRU) {
              //valid_timestamp设置为最近最少被使用的cache line的最末次访问时间。
              //valid_timestamp被初始化为(unsigned)-1，即可以看作无穷大。
              if (line->get_last_access_time() < valid_timestamp) {
                //这里的valid_timestamp是周期数，即最小的周期数具有最大的被逐出优先级，当然这个
                //变量在这里只是找具有最小周期数的cache block，最小周期数意味着离他上次使用才最
                //早，真正标识哪个cache block具有最大优先级被逐出的是valid_line。
                valid_timestamp = line->get_last_access_time();
                //标识当前cache block具有最小的执行周期数，index这个cache block应该最先被逐出。
                valid_line = index;
              }
            } else if (m_config.m_replacement_policy == FIFO) {
              if (line->get_alloc_time() < valid_timestamp) {
                //FIFO按照最早分配时间的cache block最优先被逐出。
                valid_timestamp = line->get_alloc_time();
                valid_line = index;
              }
            }
          }
        }
      } //这里是把当前set里所有的way都循环一遍，如果找到了line->m_tag == tag的块，则已经返回了
        //访问状态，如果没有找到，则也遍历了一遍所有way的cache block，找到了最优先应该被逐出和
        //替换的cache block。
    }
    //Cache访问的状态包含：
    //    HIT，HIT_RESERVED，MISS，RESERVATION_FAIL，SECTOR_MISS，MSHR_HIT六种状态。
    //抛开前面能够确定的HIT，HIT_RESERVED，SECTOR_MISS还能够判断MISS/RESERVATION_FAIL
    //两种状态是否成立。
    //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
    //被MODIFIED，来逐出；但是如果dirty的cache line的比例超过m_wr_percent（V100中配置为
    //25%），也可以不满足MODIFIED的条件。
    //all_reserved被初始化为true，是指所有cache line都没有能够逐出来为新访问提供RESERVE
    //的空间，这里一旦满足上面两个if条件，说明cache line可以被逐出来提供空间供RESERVE新访
    //问，这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明cache
    //line不可被逐出，发生RESERVATION_FAIL。
    if (all_reserved) {
      //all_reserved为true的话，表明当前set的所有way都没有cache满足被逐出的条件。因此状态
      //返回RESERVATION_FAIL，即all of the blocks in the current set have no enough 
      //space in cache to allocate on miss.
      assert(m_config.m_alloc_policy == ON_MISS);
      return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                                // on miss
    }

    //如果上面的all_reserved为false，才会到这一步，即cache line可以被逐出来为新访问提供
    //RESERVE。
    if (invalid_line != (unsigned)-1) {
      //尽管我们有LRU或者FIFO替换策略，但是最理想的情况还是优先替换整个cache block都无效
      //的块。因为这种无效的块不需要写回，能够节省带宽。
      idx = invalid_line;
    } else if (valid_line != (unsigned)-1) {
      //没有无效的块，就只能将上面按照LRU或者FIFO确定的cache block作为被逐出的块了。
      idx = valid_line;
    } else
      abort();  // if an unreserved block exists, it is either invalid or
                // replaceable

    //if (probe_mode && m_config.is_streaming()) {
    //  line_table::const_iterator i =
    //      pending_lines.find(m_config.block_addr(addr));
    //  assert(mf);
    //  if (!mf->is_write() && i != pending_lines.end()) {
    //    if (i->second != mf->get_inst().get_uid()) return SECTOR_MISS;
    //  }
    //}

    //如果上面的cache line可以被逐出来reserve新访问，则返回MISS。
    return MISS;
  }
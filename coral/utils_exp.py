


def sample_lists():
    sample_ids = ['71_2_pre',
                  '71_3_pre',

                  '72_2_pre',
                  '72_3_pre',
                  '72_4_post',
                  '72_5_post',


                  '73_2_pre',
                  '73_3_pre',
                  '73_5_post',

                  '74_2_pre',
                  '74_3_pre',
                  '74_4_post',
                  '74_5_post',

                  '76_2_pre',
                  '76_3_pre',
                  '76_4_post',
                  '76_5_post',

                  '79_2_pre',
                  '79_4_post',
                  '79_5_post',


                  '83_2_pre',

                  '84_1_pre',
                  '84_2_pre',
                  '84_3_post',
                  '84_4_post',

                  '85_2_pre',

                  '86_2_pre',
                  '86_3_pre',
                 ]
    visium_paths = [
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_71_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_71_pre/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_72_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_72_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_72_post/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_72_post/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_73_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_73_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_73_post/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_74_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_74_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_74_post/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_74_post/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_76_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_76_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_76_post/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_76_post/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_79_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_79_post/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_79_post/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_83_pre/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_84_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_84_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_84_post/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_84_post/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_85_pre/',

                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_86_pre/',
                    '/hpc/mydata/siyu.he/HCC_project/visium/cytassist_86_pre/',    

                   ]

    trans_locs = ['/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/071_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/071_3_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/072_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/072_3_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/072_4_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/072_5_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/073_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/073_3_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/073_5_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/074_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/074_3_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/074_4_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/074_5_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/076_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/076_3_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/076_4_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/076_5_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/079_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/079_4_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/079_5_tissue_positions_list.csv',


                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/083_2_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/084_1_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/084_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/084_3_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/084_4_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/085_2_tissue_positions_list.csv',

                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/086_2_tissue_positions_list.csv',
                  '/hpc/mydata/siyu.he/HCC_project/transformed_visium_loc/086_3_tissue_positions_list.csv',

                 ]

    codex_paths = ['/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg005_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg006_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg009_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg010_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg011_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg012_codex.csv',


                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg013_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg014_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg016_codex.csv',


                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg017_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg018_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg019_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg020_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg001_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg002_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg003_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg004_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg029_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg031_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c001_v001_r001_reg032_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg005_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg008_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg009_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg010_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg011_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg012_codex.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg015_codex.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_seg/FinalLiv-27_c002_v001_r001_reg016_codex.csv',
                 ]


    type_paths = [
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg005.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg006.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg009.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg010.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg011.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg012.cell_types.csv',


                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg013.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg014.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg016.cell_types.csv',


                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg017.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg018.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg019.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg020.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg001.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg002.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg003.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg004.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg029.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg031.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c001_v001_r001_reg032.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg005.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg008.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg009.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg010.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg011.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg012.cell_types.csv',

                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg015.cell_types.csv',
                   '/hpc/mydata/siyu.he/HCC_project/codex_cell_type/FinalLiv-27_c002_v001_r001_reg016.cell_types.csv',
                 ]
    
    
    return sample_ids, visium_paths, trans_locs, codex_paths, type_paths

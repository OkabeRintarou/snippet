set nu

syntax on
filetype plugin indent on

set tabstop=4
set shiftwidth=4
autocmd FileType rust set expandtab

filetype off                  " required

let mapleader=','

inoremap jj <Esc>`^

inoremap <leader>w <Esc>:w<cr>
noremap <leader>w :w<cr>

noremap <C-h> <C-w>h
noremap <C-j> <C-w>j
noremap <C-k> <C-w>k
noremap <C-l> <C-w>l
"==================== Begin Plugin ====================
call plug#begin()

Plug 'preservim/nerdtree'
Plug 'vim-airline/vim-airline'
Plug 'ludovicchabant/vim-gutentags'
Plug 'skywind3000/gutentags_plus'
Plug 'skywind3000/vim-preview'
Plug 'Yggdroot/LeaderF', { 'do': ':LeaderfInstallCExtension' }

" Elixir
Plug 'elixir-editors/vim-elixir'
Plug 'dense-analysis/ale'

call plug#end()
"==================== End Plugin ====================

nnoremap <leader>v :NERDTreeFind<CR>
nnoremap <leader>g :NERDTreeToggle<CR>

"==================== Begin ALE ==================
set completeopt=menu,menuone,preview,noselect,noinsert
let g:ale_completion_enabled = 1
let g:ale_linters_explicit = 1
let g:ale_linters = {}

"let g:airline#extensions#ale#enabled = 1
let g:ale_linters.cpp = ['clang++', 'g++']
let g:ale_linters.c = ['clang']

let g:ale_c_cc_options = '-Wall -O2 -std=c99'
let g:ale_cpp_cc_options = '-std=c++20 -Wall -O2'

let g:ale_linters.elixir = ['elixir-ls']
let g:ale_elixir_elixir_ls_release = expand("~/soft/elixir-ls/rel")
let g:ale_elixir_elixir_ls_config = {'elixirLS': {'dialyzerEnabled': v:false}}

"==================== End ALE ====================

"==================== Begin gutentags ====================
set tags=./.tags;,.tags

" gutentags 搜索工程目录的标志，当前文件路径向上递归直到碰到这些文件/目录名
let g:gutentags_project_root = ['.root', '.svn', '.git', '.hg', '.project']

" 所生成的数据文件的名称
let g:gutentags_ctags_tagfile = '.tags'

" 同时开启 ctags 和 gtags 支持：
let g:gutentags_modules = []
if executable('ctags')
	let g:gutentags_modules += ['ctags']
endif
if executable('gtags-cscope') && executable('gtags')
	let g:gutentags_modules += ['gtags_cscope']
endif

" change focus to quickfix window after search (optional).
let g:gutentags_plus_switch = 1

" 将自动生成的 ctags/gtags 文件全部放入 ~/.cache/tags 目录中，避免污染工程目录
let g:gutentags_cache_dir = expand('~/.cache/tags')

" 配置 ctags 的参数，老的 Exuberant-ctags 不能有 --extra=+q，注意
let g:gutentags_ctags_extra_args = ['--fields=+niazS', '--extra=+q']
let g:gutentags_ctags_extra_args += ['--c++-kinds=+px']
let g:gutentags_ctags_extra_args += ['--c-kinds=+px']

" 如果使用 universal ctags 需要增加下面一行，老的 Exuberant-ctags 不能加下一行
let g:gutentags_ctags_extra_args += ['--output-format=e-ctags']

" 禁用 gutentags 自动加载 gtags 数据库的行为
let g:gutentags_auto_add_gtags_cscope = 0
"==================== End gutentags ====================

noremap <m-u> :PreviewScroll -1<cr>
noremap <m-d> :PreviewScroll +1<cr>
inoremap <m-u> <c-\><c-o>:PreviewScroll -1<cr>
inoremap <m-d> <c-\><c-o>:PreviewScroll +1<cr>

autocmd FileType qf nnoremap <silent><buffer> p :PreviewQuickfix<cr>
autocmd FileType qf nnoremap <silent><buffer> P :PreviewClose<cr>

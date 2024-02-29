"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[1697],{7919:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>c,contentTitle:()=>l,default:()=>h,frontMatter:()=>i,metadata:()=>a,toc:()=>d});var t=s(4848),r=s(8453);const i={sidebar_position:2},l="Neural networks and deep learning basics",a={id:"lectures/nn-basics",title:"Neural networks and deep learning basics",description:"\u8bfe\u7a0b\u5927\u7eb2",source:"@site/docs/lectures/2-nn-basics.md",sourceDirName:"lectures",slug:"/lectures/nn-basics",permalink:"/cs2916/docs/lectures/nn-basics",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/lectures/2-nn-basics.md",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2},sidebar:"tutorialSidebar",previous:{title:"Why LLMs?",permalink:"/cs2916/docs/lectures/why-llms"},next:{title:"Language Models and Representation Learning",permalink:"/cs2916/docs/lectures/lms"}},c={},d=[{value:"\u8bfe\u7a0b\u5927\u7eb2",id:"\u8bfe\u7a0b\u5927\u7eb2",level:2},{value:"\u7ec3\u4e60",id:"\u7ec3\u4e60",level:2},{value:"\u63a8\u8350\u9605\u8bfb\u6750\u6599",id:"\u63a8\u8350\u9605\u8bfb\u6750\u6599",level:2}];function o(e){const n={a:"a",h1:"h1",h2:"h2",li:"li",ul:"ul",...(0,r.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h1,{id:"neural-networks-and-deep-learning-basics",children:"Neural networks and deep learning basics"}),"\n",(0,t.jsx)(n.h2,{id:"\u8bfe\u7a0b\u5927\u7eb2",children:"\u8bfe\u7a0b\u5927\u7eb2"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["\u795e\u7ecf\u7f51\u7edc\u57fa\u7840\u6982\u5ff5 [",(0,t.jsx)(n.a,{target:"_blank","data-noBrokenLinkCheck":!0,href:s(2740).A+"",children:"\u8bfe\u4ef6"}),"]"]}),"\n",(0,t.jsxs)(n.li,{children:["\u5faa\u73af\u795e\u7ecf\u7f51\u7edc [",(0,t.jsx)(n.a,{target:"_blank","data-noBrokenLinkCheck":!0,href:s(4539).A+"",children:"\u8bfe\u4ef6"}),"]"]}),"\n",(0,t.jsxs)(n.li,{children:["\u5377\u79ef\u795e\u7ecf\u7f51\u7edc [",(0,t.jsx)(n.a,{target:"_blank","data-noBrokenLinkCheck":!0,href:s(984).A+"",children:"\u8bfe\u4ef6"}),"]"]}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"\u7ec3\u4e60",children:"\u7ec3\u4e60"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:(0,t.jsx)(n.a,{target:"_blank","data-noBrokenLinkCheck":!0,href:s(5762).A+"",children:"Feedforward neural network"})}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"\u63a8\u8350\u9605\u8bfb\u6750\u6599",children:"\u63a8\u8350\u9605\u8bfb\u6750\u6599"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["[\u8bfe\u4ef6]",(0,t.jsx)(n.a,{href:"https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture05-rnnlm.pdf",children:"CS224n (Lecture06)"})]}),"\n",(0,t.jsxs)(n.li,{children:["[\u8bfe\u4ef6]",(0,t.jsx)(n.a,{href:"https://www.phontron.com/class/nn4nlp2021/schedule/rnn.html",children:"CS11747 (Lecture05)"})]}),"\n",(0,t.jsxs)(n.li,{children:["[\u8bba\u6587]",(0,t.jsx)(n.a,{href:"https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf",children:"RNN"})]}),"\n",(0,t.jsxs)(n.li,{children:["[\u8bba\u6587]",(0,t.jsx)(n.a,{href:"https://www.bioinf.jku.at/publications/older/2604.pdf",children:"LSTM"})]}),"\n",(0,t.jsxs)(n.li,{children:["[\u8bba\u6587]",(0,t.jsx)(n.a,{href:"https://arxiv.org/abs/1408.5882",children:"CNN"})]}),"\n",(0,t.jsxs)(n.li,{children:["[\u4e66\u7c4d]",(0,t.jsx)(n.a,{href:"https://nndl.github.io/nndl-book.pdf",children:"\u6df1\u5ea6\u7f51\u7edc\u4e0e\u6df1\u5ea6\u5b66\u4e60\uff08\u7b2c\u56db\u7ae0\u8282\uff09"})]}),"\n",(0,t.jsxs)(n.li,{children:["[\u8bfe\u7a0b]",(0,t.jsx)(n.a,{href:"https://v.youku.com/v_show/id_XNjU1MzY4ODQ4.html?f=27111126&spm=a2hje.13141534.1_2.d_1_3&scm=20140719.apircmd.240116.video_XNjU1MzY4ODQ4",children:"\u5434\u7acb\u5fb7\u8001\u5e08\u6df1\u5ea6\u5b66\u4e60\u8bfe\u7a0b"})]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(o,{...e})}):o(e)}},5762:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/files/nn_bp-bee2999c27c7ab32e0788f422816a8b6.py"},2740:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/files/lecture02-part1-70729cead9a6a488d6422f5ca055fd14.pdf"},4539:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/files/lecture02-part2-9ece7599e353a686a0d380bbbf3cb840.pdf"},984:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/files/lecture02-part3-7d5a33f56fd46df7c4b861108fdfb749.pptx"},8453:(e,n,s)=>{s.d(n,{R:()=>l,x:()=>a});var t=s(6540);const r={},i=t.createContext(r);function l(e){const n=t.useContext(i);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:l(e.components),t.createElement(i.Provider,{value:n},e.children)}}}]);

import stevens from '../../public/icons/companies/stevens.jpeg';
import tiaa from '../../public/icons/companies/tiaa.svg';
import robocon from '../../public/icons/companies/robocon.jpeg';
import cc from '../../public/icons/companies/CC.jpeg';
import keosworld from '../../public/icons/tech/keosworld.jpeg';


// Projects
import simplilearn from '../../public/icons/companies/simplilearn.png';
import codingninjas from '../../public/icons/companies/codingninjas.svg';

// Skills
// Languages
import html from '../../public/icons/tech/html.svg';
import javascript from '../../public/icons/tech/javascript.svg';
import typescript from '../../public/icons/tech/typescript.svg';
import python from '../../public/icons/tech/python.svg';
import golang from '../../public/icons/tech/golang.svg';
import csharp from '../../public/icons/tech/csharp.svg';
import cpp from '../../public/icons/tech/cpp.svg';
import sql from '../../public/icons/tech/sql.svg';
import shell from '../../public/icons/tech/bash_shell.svg';
import java from '../../public/icons/tech/java.svg';
import solidity from '../../public/icons/tech/solidity.svg';
// Frontend
import css from '../../public/icons/tech/css.svg';
import react from '../../public/icons/tech/react.svg';
import vuejs from '../../public/icons/tech/vuejs.svg';
import angular from '../../public/icons/tech/angular.svg';
import javafx from '../../public/icons/tech/javafx.svg';
// Backend
import expressjs from '../../public/icons/tech/expressjs.svg';
import flask from '../../public/icons/tech/flask.svg';
import django from '../../public/icons/tech/django.svg';
import spring from '../../public/icons/tech/spring.svg';
import nodejs from '../../public/icons/tech/nodejs.svg';
import crow from '../../public/icons/tech/crowcpp.svg';
// Databases
import nextjs from '../../public/icons/tech/nextjs.svg';
import firebase from '../../public/icons/tech/firebase.svg';
import authjs from '../../public/icons/tech/authjs.svg';
import postgresql from '../../public/icons/tech/postgresql.svg';
import mysql from '../../public/icons/tech/mysql.svg';
import firestore from '../../public/icons/tech/firestore.svg';
import mongodb from '../../public/icons/tech/mongodb.svg';
import couchdb from '../../public/icons/tech/couchdb.svg';
import elasticsearch from '../../public/icons/tech/elasticsearch.svg';
import neo4j from '../../public/icons/tech/neo4j.svg';
import snowflake from '../../public/icons/tech/snowflake.svg';
import spark from '../../public/icons/tech/spark.png';
// Infrastructure
import aws from '../../public/icons/tech/aws.svg';
import azure from '../../public/icons/tech/azure.svg';
import linux from '../../public/icons/tech/linux.svg';
import googlecloudplatform from '../../public/icons/tech/googlecloudplatform.svg';
import docker from '../../public/icons/tech/docker.svg';
import kubernetes from '../../public/icons/tech/kubernets.svg';
import github from '../../public/icons/tech/github.svg';
import gitlab from '../../public/icons/tech/gitlab.svg';
import ethereum from '../../public/icons/tech/ethereum.svg';
// Tools
import vscode from '../../public/icons/tech/vscode.svg';
import intellij from '../../public/icons/tech/intellij-idea.svg';
import graphql from '../../public/icons/tech/graphql.svg';
import airflow from '../../public/icons/tech/airflow.svg';
import jira from '../../public/icons/tech/jira.svg';
import chatgpt from '../../public/icons/tech/chatgpt.svg';
import git from '../../public/icons/tech/git.svg';
// Machine Learning
import pandas from '../../public/icons/tech/pandas.svg';
import numpy from '../../public/icons/tech/numpy.svg';
import keras from '../../public/icons/tech/Keras.svg';
import matplotlib from '../../public/icons/tech/matplotlib.svg';
import sklearn from '../../public/icons/tech/sklearn.svg';
import tensorflow from '../../public/icons/tech/tensorflow.svg';
import pytorch from '../../public/icons/tech/pytorch.svg';
// GenAI
import openai from '../../public/icons/tech/openai.svg';
import langchain from '../../public/icons/tech/langchain.svg';
import pinecone from '../../public/icons/tech/pinecone.svg';
import chromadb from '../../public/icons/tech/chromaDB.svg';
import mistral from '../../public/icons/tech/mistral.svg';
import llama2 from '../../public/icons/tech/llama2.svg';

export const navLinks = [
  {
    id: 'about',
    title: 'About'
  },
  {
    id: 'work',
    title: 'Work'
  },
  {
    id: 'project',
    title: 'Projects'
  },
  {
    id: 'achievements',
    title: 'Achievements'
  },
  {
    id: 'resume',
    title: 'Resume',
    external_link: 'https://drive.google.com/file/d/1kSHiqvbroWuPKDrRBx43KA8PPJUKA2j_/view?usp=drive_link'
  }
];

export const services = [
  {
    title: 'LANGUAGES',
    icons: [
      {
        name: 'Python',
        icon: python,
        link: 'https://www.python.org/',
      },
      {
        name: 'JavaScript',
        icon: javascript,
        link: 'https://developer.mozilla.org/docs/Web/javascript/',
      },
      {
        name: 'TypeScript',
        icon: typescript,
        link: 'https://www.typescriptlang.org/',
      },
      {
        name: 'Java',
        icon: java,
        link: 'https://www.java.com/',
      },
      {
        name: 'SQL',
        icon: sql,
        link: 'https://sql.org/',
      },
      {
        name: 'C#',
        icon: csharp,
        link: 'https://learn.microsoft.com/en-us/dotnet/csharp/',
      },
      {
        name: 'C++',
        icon: cpp,
        link: 'https://isocpp.org/',
      },
      {
        name: 'HTML',
        icon: html,
        link: 'https://developer.mozilla.org/docs/Web/HTML/',
      },
      {
        name: 'Shell',
        icon: shell,
        link: 'https://www.gnu.org/software/bash/',
      }
    ],
  },
  {
    title: 'FRONTEND',
    icons: [
      {
        name: ' CSS',
        icon: css,
        link: 'https://css.com/',
      },
      {
        name: 'React',
        icon: react,
        link: 'https://react.dev/',
      },
      {
        name: 'Angular',
        icon: angular,
        link: 'https://angular.dev/',
      }
    ],
  },
  {
    title: 'BACKEND',
    icons: [
      {
        name: 'Flask',
        icon: flask,
        link: 'https://flask.palletsprojects.com/en/3.0.x/',
      },
      {
        name: 'Django',
        icon: django,
        link: 'https://www.djangoproject.com/',
      },
      {
        name: 'Node.js',
        icon: nodejs,
        link: 'https://nodejs.org/en',
      },
      {
        name: 'Next.js',
        icon: nextjs,
        link: 'https://nextjs.org/',
      },
      {
        name: 'Express.js',
        icon: expressjs,
        link: 'https://expressjs.com/',
      },
      {
        name: 'Firebase',
        icon: firebase,
        link: 'https://firebase.google.com/',
      },
      {
        name: 'Auth.js',
        icon: authjs,
        link: 'https://authjs.dev/',
      },
    ],
  },
  {
    title: 'DATABASES',
    icons: [
      {
        name: 'PostgreSQL',
        icon: postgresql,
        link: 'https://www.postgresql.org/',
      },
      {
        name: 'MySQL',
        icon: mysql,
        link: 'https://www.mysql.com/',
      },
      {
        name: 'MongoDB',
        icon: mongodb,
        link: 'https://www.mongodb.com/',
      },
      {
        name: 'Neo4j',
        icon: neo4j,
        link: 'https://neo4j.com/',
      },
      {
        name: 'Firestore',
        icon: firestore,
        link: 'https://cloud.google.com/firestore/',
      },
      {
        name: 'PySpark',
        icon: spark,
        link: 'https://pypi.org/project/pyspark/',
      },
    ],
  },
  {
    title: 'INFRASTRUCTURE',
    icons: [
      {
        name: 'AWS Cloud Services',
        icon: aws,
        link: 'https://aws.amazon.com/',
      },
      {
        name: 'Azure Cloud Services',
        icon: azure,
        link: 'https://azure.microsoft.com/en-us',
      },
      {
        name: 'Google Cloud Platform',
        icon: googlecloudplatform,
        link: 'https://cloud.google.com/',
      },
      {
        name: 'Docker',
        icon: docker,
        link: 'https://www.docker.com/',
      },
      {
        name: 'GitHub',
        icon: github,
        link: 'https://docs.github.com/',
      },
      {
        name: 'GitLab',
        icon: gitlab,
        link: 'https://docs.gitlab.com/',
      },
      {
        name: 'Linux',
        icon: linux,
        link: 'https://www.linux.org/',
      }
    ],
  },
  {
    title: 'TOOLS',
    icons: [
      {
        name: 'Visual Studio Code',
        icon: vscode,
        link: 'https://code.visualstudio.com/',
      },
      {
        name: 'IntelliJ IDEA',
        icon: intellij,
        link: 'https://www.jetbrains.com/idea/',
      },
      {
        name: 'GraphQL',
        icon: graphql,
        link: 'https://graphql.org/',
      },
      {
        name: 'Airflow',
        icon: airflow,
        link: 'https://airflow.apache.org/',
      },
      {
        name: 'ChatGPT',
        icon: chatgpt,
        link: 'https://openai.com/blog/chatgpt/',
      },
      {
        name: 'Git',
        icon: git,
        link: 'https://git-scm.com/',
      },
    ],
  },
  {
  title: 'Machine Learning',
  icons: [
    {
      name: 'Pandas',
      icon: pandas,
      link: 'https://pandas.pydata.org/',
    },
    {
      name: 'Numpy',
      icon: numpy,
      link: 'https://numpy.org/',
    },
    {
      name: 'Matplotlib',
      icon: matplotlib,
      link: 'https://matplotlib.org/',
    },
    {
      name: 'Tensorflow',
      icon: tensorflow,
      link: 'https://www.tensorflow.org/',
    },
    {
      name: 'PyTorch',
      icon: pytorch,
      link: 'https://pytorch.org/',
    },
    {
      name: 'Sci-kit Learn',
      icon: sklearn,
      link: 'https://scikit-learn.org/stable/',
    },
    {
      name: 'Keras',
      icon: keras,
      link: 'https://keras.io/',
    },
  ],
},
{
  title: 'Generative AI',
  icons: [
    {
      name: 'OpenAI',
      icon: openai,
      link: 'https://openai.com',
    },
    {
      name: 'LLama2',
      icon: llama2,
      link: 'https://llama.meta.com/llama2/',
    },
    {
      name: 'Mistral',
      icon: mistral,
      link: 'https://mistral.ai/',
    },
    {
      name: 'Langchain',
      icon: langchain,
      link: 'https://www.langchain.com/',
    },
    {
      name: 'ChromaDB',
      icon: chromadb,
      link: 'https://www.trychroma.com/',
    },
    {
      name: 'Pinecone',
      icon: pinecone,
      link: 'https://www.pinecone.io/',
    },
  ],
},
];

export const experiences = [
  {
    title: 'Teaching Assistant – Data Structures',
    companyName: 'Stevens Institute of Technology',
    icon: stevens,
    date: 'Jan 2025 – Present',
    points: [
      'Assist in teaching core Data Structures concepts in Java including arrays, linked lists, stacks, queues, trees, heaps, graphs, and hash tables.',
      'Conduct lab sessions and walkthroughs to help students implement data structures and understand their time and space complexities.',
      'Hold weekly office hours to resolve doubts, review logic, and guide students in debugging and improving their solutions.',
      'Grade assignments, coding tasks, and exams with detailed feedback to help students improve their understanding.',
      'Collaborate with the professor to maintain course flow, assist during exams, and ensure a smooth learning experience.'
    ],
  },
  {
    title: 'Software Developer Analyst',
    companyName: 'TIAA',
    icon: tiaa,
    date: 'Aug 2021 – Aug 2023',
    points: [
      'Automated production support tasks using Python, reducing manual efforts by 70% and enhancing production stability.',
      'Worked on Proof-of-Concepts (PoCs) with deep learning models to classify and assign issue tickets, achieving 94% classification accuracy and significantly improving turnaround time.',
      'Gained hands-on experience across full-stack tech including Java, Angular, and Hadoop; used Splunk and Dynatrace for log analysis and monitoring.',
      'Progressed from Trainee Support Analyst to Software Developer Analyst within a year through consistent delivery and initiative.',
      'Collaborated with cross-functional teams to enhance performance and stability of TIAA’s key financial products and services.'
    ],
  },
  
  {
    title: 'Vice-Captain & Embedded Systems Developer',
    companyName: 'Robocon Tech Team',
    icon: robocon,
    date: 'Aug 2019 – July 2021',
    points: [
      'Led a 20-member interdisciplinary team to design and develop two semi-autonomous robots for the ABU-Robocon competition.',
      'Wrote embedded C codes to interface with PS3 controllers, enabling multidirectional bot movement with real-time response.',
      'Integrated physics, electronics, and control systems to reach All India Rank 9 in stage 2 of Robocon.',
      'Managed team communication, hardware testing, and final deployment strategy under high-pressure deadlines.'
    ],
  },
  {
    title: 'AI Intern',
    companyName: 'Cloud Counselage',
    icon: cc,
    date: 'Aug 2019 – July 2021',
    points: [
      'Led a team of over 15 interns during a two-month internship, managing projects focused on Machine Learning, Natural Language Processing, and Software Testing. Ensured effective collaboration and timely project completion while fostering a productive learning environment.',
      'Mentored and guided interns in core AI concepts, including feature engineering, model evaluation, and NLP pipelines, cultivating technical expertise and improving team performance throughout the internship period.'
    ],
  },
  
  
  
];


export const projects = [
  {
    name: 'KeosWorld — E-commerce Platform',
    description:
      "An end-to-end e-commerce web application where users can browse products, manage carts, place orders, and admins can manage inventory. Built with React, Redux, Node.js, Express, MongoDB, and Firebase for auth.",
    techstack: [
      {
        name: 'JavaScript',
        icon: javascript,
        link: 'https://developer.mozilla.org/docs/Web/javascript/',
      },
      {
        name: 'Express.js',
        icon: expressjs,
        link: 'https://expressjs.com/',
      },
      {
        name: 'MySQL',
        icon: mysql,
        link: 'https://mysql.org/',
      },
      {
        name: 'React',
        icon: react,
        link: 'https://react.dev/',
      },
      {
        name: 'HTML',
        icon: html,
        link: 'https://developer.mozilla.org/docs/Web/HTML/',
      },
      {
        name: 'CSS',
        icon: css,
        link: 'https://developer.mozilla.org/docs/Web/CSS/',
      },
      {
        name: 'Firebase',
        icon: firebase,
        link: 'https://firebase.google.com/',
  
      }
    ],
    image: keosworld,
    prodLink: 'https://www.youtube.com/watch?v=JZg64Wgq-54',
    srcLink: 'https://github.com/nitinchaube/KeosWorld/blob/main/README.md',
  }
];

export const techUsed = [
  {
    name: 'CSS',
    icon: css,
    link: 'https://developer.mozilla.org/docs/Web/CSS/',
  },
  {
    name: ' CSS',
    icon: css,
    link: 'https://css.com/',
  },
  {
    name: 'TypeScript',
    icon: typescript,
    link: 'https://www.typescriptlang.org/',
  },
  {
    name: 'React',
    icon: react,
    link: 'https://react.dev/',
  },
  {
    name: 'Next.js',
    icon: nextjs,
    link: 'https://nextjs.org/',
  },
  {
    name: 'GitHub',
    icon: github,
    link: 'https://docs.github.com/',
  },
];
export const achievements = [
  
  {
    achievement:
      "Conducted research on CrisisLex and CrisisNLP datasets, consolidating over 500,000 crisis-related tweets to build an AI-driven disaster informatics model for accurate crisis classification. ",
    name: 'FLOOD_KNOW - Information Graph ',
    designation: 'Research Paper',
    image: tensorflow,
    link: 'https://tijer.org/tijer/papers/TIJER2309075.pdf'
  },
  {
    achievement:
      "I completed the Artificial Intelligence Engineer Master’s Program by Simplilearn in collaboration with IBM, where I gained in-depth knowledge of core AI concepts and their real-world business applications. The program covered a broad spectrum of topics including machine learning, deep learning, natural language processing, and neural networks, equipping me with both theoretical understanding and hands-on experience in building AI-driven solutions.",
    name: 'Artificial Intelligence Engineer ',
    designation: 'Certification',
    image: simplilearn,
    link: 'https://success.simplilearn.com/e517b81b-00dd-468d-9859-84b5dd18e095#acc.mgAVQ9Mu'
  },
  {
    achievement:
      "I completed the Competitive Programming certification course from Coding Ninjas, where I developed strong problem-solving skills and a deep understanding of data structures and algorithms. Through consistent practice and hands-on challenges, I sharpened my ability to tackle complex algorithmic problems under time constraints",
    name: 'Competitive Programming ',
    designation: 'Certification',
    image: codingninjas,
    link: 'https://files.codingninjas.in/certificate11738949934970267154ef6c7eafb2272ed78d5.pdf'
  },

  
];


# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import shutil
import glob
import json
import urllib.request
import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

def main():
    temp_folder_repo = 'essay_repo'
    temp_folder_html = 'essay_html'
    os.makedirs(temp_folder_repo, exist_ok=True)
    os.makedirs(temp_folder_html, exist_ok=True)

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_tables = True
    h.escape_all = True
    h.reference_links = False
    h.mark_code = False

    urls = PAUL_GRAHAM_ESSAYS


    for url in tqdm(urls):
        if '.html' in url:
            filename = url.split('/')[-1].replace('.html', '.txt')        
            try:
                with urllib.request.urlopen(url) as website:
                    content = website.read().decode("unicode_escape", "utf-8")
                    soup = BeautifulSoup(content, 'html.parser')
                    specific_tag = soup.find('font')
                    parsed = h.handle(str(specific_tag))
                    
                    with open(os.path.join(temp_folder_html, filename), 'w') as file:
                        file.write(parsed)
            
            except Exception as e:
                print(f"Fail download {filename}, ({e})")

        else:
            filename = url.split('/')[-1]
            try:
                with urllib.request.urlopen(url) as website:
                    content = website.read().decode('utf-8')
                
                with open(os.path.join(temp_folder_repo, filename), 'w') as file:
                    file.write(content)
                        
            except Exception as e:
                print(f"Fail download {filename}, ({e})")

    files_repo = sorted(glob.glob(os.path.join(temp_folder_repo,'*.txt')))
    files_html = sorted(glob.glob(os.path.join(temp_folder_html,'*.txt')))
    print(f'Download {len(files_repo)} essays from `https://github.com/gkamradt/LLMTest_NeedleInAHaystack/`') 
    print(f'Download {len(files_html)} essays from `http://www.paulgraham.com/`') 

    text = ""
    for file in files_repo + files_html:
        with open(file, 'r') as f:
            text += f.read()

    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_data')
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'PaulGrahamEssays.json'), 'w') as f:
        json.dump({"text": text}, f)


    shutil.rmtree(temp_folder_repo)
    shutil.rmtree(temp_folder_html)


PAUL_GRAHAM_ESSAYS = [
    "http://www.paulgraham.com/13sentences.html",
    "http://www.paulgraham.com/5founders.html",
    "http://www.paulgraham.com/6631327.html",
    "http://www.paulgraham.com/95.html",
    "http://www.paulgraham.com/ace.html",
    "http://www.paulgraham.com/airbnb.html",
    "http://www.paulgraham.com/airbnbs.html",
    "http://www.paulgraham.com/alien.html",
    "http://www.paulgraham.com/altair.html",
    "http://www.paulgraham.com/ambitious.html",
    "http://www.paulgraham.com/america.html",
    "http://www.paulgraham.com/angelinvesting.html",
    "http://www.paulgraham.com/artistsship.html",
    "http://www.paulgraham.com/badeconomy.html",
    "http://www.paulgraham.com/better.html",
    "http://www.paulgraham.com/bronze.html",
    "http://www.paulgraham.com/bubble.html",
    "http://www.paulgraham.com/charisma.html",
    "http://www.paulgraham.com/cities.html",
    "http://www.paulgraham.com/college.html",
    "http://www.paulgraham.com/colleges.html",
    "http://www.paulgraham.com/conformism.html",
    "http://www.paulgraham.com/control.html",
    "http://www.paulgraham.com/convergence.html",
    "http://www.paulgraham.com/convince.html",
    "http://www.paulgraham.com/cred.html",
    "http://www.paulgraham.com/credentials.html",
    "http://www.paulgraham.com/determination.html",
    "http://www.paulgraham.com/die.html",
    "http://www.paulgraham.com/disagree.html",
    "http://www.paulgraham.com/disc.html",
    "http://www.paulgraham.com/discover.html",
    "http://www.paulgraham.com/distraction.html",
    "http://www.paulgraham.com/divergence.html",
    "http://www.paulgraham.com/donate.html",
    "http://www.paulgraham.com/ds.html",
    "http://www.paulgraham.com/early.html",
    "http://www.paulgraham.com/earnest.html",
    "http://www.paulgraham.com/equity.html",
    "http://www.paulgraham.com/essay.html",
    "http://www.paulgraham.com/ffb.html",
    "http://www.paulgraham.com/fh.html",
    "http://www.paulgraham.com/fix.html",
    "http://www.paulgraham.com/fn.html",
    "http://www.paulgraham.com/foundersatwork.html",
    "http://www.paulgraham.com/fp.html",
    "http://www.paulgraham.com/fr.html",
    "http://www.paulgraham.com/fundraising.html",
    "http://www.paulgraham.com/future.html",
    "http://www.paulgraham.com/genius.html",
    "http://www.paulgraham.com/getideas.html",
    "http://www.paulgraham.com/good.html",
    "http://www.paulgraham.com/goodart.html",
    "http://www.paulgraham.com/googles.html",
    "http://www.paulgraham.com/greatwork.html",
    "http://www.paulgraham.com/growth.html",
    "http://www.paulgraham.com/guidetoinvestors.html",
    "http://www.paulgraham.com/hackernews.html",
    "http://www.paulgraham.com/head.html",
    "http://www.paulgraham.com/herd.html",
    "http://www.paulgraham.com/heresy.html",
    "http://www.paulgraham.com/heroes.html",
    "http://www.paulgraham.com/highres.html",
    "http://www.paulgraham.com/hiresfund.html",
    "http://www.paulgraham.com/hiring.html",
    "http://www.paulgraham.com/hp.html",
    "http://www.paulgraham.com/hs.html",
    "http://www.paulgraham.com/hundred.html",
    "http://www.paulgraham.com/hw.html",
    "http://www.paulgraham.com/hwh.html",
    "http://www.paulgraham.com/icad.html",
    "http://www.paulgraham.com/ideas.html",
    "http://www.paulgraham.com/identity.html",
    "http://www.paulgraham.com/ineq.html",
    "http://www.paulgraham.com/inequality.html",
    "http://www.paulgraham.com/investors.html",
    "http://www.paulgraham.com/invtrend.html",
    "http://www.paulgraham.com/javacover.html",
    "http://www.paulgraham.com/jessica.html",
    "http://www.paulgraham.com/judgement.html",
    "http://www.paulgraham.com/kate.html",
    "http://www.paulgraham.com/kids.html",
    "http://www.paulgraham.com/ladder.html",
    "http://www.paulgraham.com/lesson.html",
    "http://www.paulgraham.com/lies.html",
    "http://www.paulgraham.com/lwba.html",
    "http://www.paulgraham.com/mac.html",
    "http://www.paulgraham.com/makersschedule.html",
    "http://www.paulgraham.com/marginal.html",
    "http://www.paulgraham.com/maybe.html",
    "http://www.paulgraham.com/mean.html",
    "http://www.paulgraham.com/microsoft.html",
    "http://www.paulgraham.com/mit.html",
    "http://www.paulgraham.com/name.html",
    "http://www.paulgraham.com/nerds.html",
    "http://www.paulgraham.com/newthings.html",
    "http://www.paulgraham.com/noob.html",
    "http://www.paulgraham.com/noop.html",
    "http://www.paulgraham.com/notnot.html",
    "http://www.paulgraham.com/nov.html",
    "http://www.paulgraham.com/nthings.html",
    "http://www.paulgraham.com/opensource.html",
    "http://www.paulgraham.com/organic.html",
    "http://www.paulgraham.com/orth.html",
    "http://www.paulgraham.com/own.html",
    "http://www.paulgraham.com/patentpledge.html",
    "http://www.paulgraham.com/pgh.html",
    "http://www.paulgraham.com/pinch.html",
    "http://www.paulgraham.com/polls.html",
    "http://www.paulgraham.com/power.html",
    "http://www.paulgraham.com/prcmc.html",
    "http://www.paulgraham.com/procrastination.html",
    "http://www.paulgraham.com/progbot.html",
    "http://www.paulgraham.com/prop62.html",
    "http://www.paulgraham.com/property.html",
    "http://www.paulgraham.com/publishing.html",
    "http://www.paulgraham.com/pypar.html",
    "http://www.paulgraham.com/ramenprofitable.html",
    "http://www.paulgraham.com/randomness.html",
    "http://www.paulgraham.com/re.html",
    "http://www.paulgraham.com/read.html",
    "http://www.paulgraham.com/real.html",
    "http://www.paulgraham.com/really.html",
    "http://www.paulgraham.com/relres.html",
    "http://www.paulgraham.com/revolution.html",
    "http://www.paulgraham.com/richnow.html",
    "http://www.paulgraham.com/road.html",
    "http://www.paulgraham.com/ronco.html",
    "http://www.paulgraham.com/safe.html",
    "http://www.paulgraham.com/say.html",
    "http://www.paulgraham.com/schlep.html",
    "http://www.paulgraham.com/seesv.html",
    "http://www.paulgraham.com/segway.html",
    "http://www.paulgraham.com/selfindulgence.html",
    "http://www.paulgraham.com/sfp.html",
    "http://www.paulgraham.com/simply.html",
    "http://www.paulgraham.com/smart.html",
    "http://www.paulgraham.com/softwarepatents.html",
    "http://www.paulgraham.com/spam.html",
    "http://www.paulgraham.com/speak.html",
    "http://www.paulgraham.com/start.html",
    "http://www.paulgraham.com/startupfunding.html",
    "http://www.paulgraham.com/startuphubs.html",
    "http://www.paulgraham.com/startupideas.html",
    "http://www.paulgraham.com/startupmistakes.html",
    "http://www.paulgraham.com/stuff.html",
    "http://www.paulgraham.com/superlinear.html",
    "http://www.paulgraham.com/swan.html",
    "http://www.paulgraham.com/tablets.html",
    "http://www.paulgraham.com/talk.html",
    "http://www.paulgraham.com/taste.html",
    "http://www.paulgraham.com/think.html",
    "http://www.paulgraham.com/top.html",
    "http://www.paulgraham.com/trolls.html",
    "http://www.paulgraham.com/twitter.html",
    "http://www.paulgraham.com/usa.html",
    "http://www.paulgraham.com/users.html",
    "http://www.paulgraham.com/venturecapital.html",
    "http://www.paulgraham.com/wealth.html",
    "http://www.paulgraham.com/webstartups.html",
    "http://www.paulgraham.com/whyyc.html",
    "http://www.paulgraham.com/word.html",
    "http://www.paulgraham.com/words.html",
    "http://www.paulgraham.com/work.html",
    "http://www.paulgraham.com/writing44.html",
    "http://www.paulgraham.com/wtax.html",
    "http://www.paulgraham.com/yahoo.html",
    "http://www.paulgraham.com/ycombinator.html",
    "http://www.paulgraham.com/ycstart.html",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/addiction.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/aord.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/apple.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/avg.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/before.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/bias.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/boss.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/copy.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/corpdev.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/desres.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/diff.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/ecw.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/founders.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/foundervisa.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/gap.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/gba.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/gh.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/goodtaste.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/hubs.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/iflisp.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/island.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/know.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/langdes.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/laundry.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/love.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/mod.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/newideas.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/nft.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/philosophy.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/popular.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/pow.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/rootsoflisp.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/rss.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/siliconvalley.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/startuplessons.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/submarine.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/sun.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/superangels.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/todo.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/unions.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/useful.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/vb.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/vcsqueeze.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/vw.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/want.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/web20.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/weird.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/wisdom.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/worked.txt"
]

if __name__ == "__main__":
    main()
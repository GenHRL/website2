document.addEventListener('DOMContentLoaded', function () {
    // Target an element by its h4 or h6 tag, then toggle the next sibling element that is a details-container
    // This is a common pattern for accordions where the clickable header is followed by the content to show/hide.

    // Handle Level 2 skills (h4 headers)
    const level2SkillHeaders = document.querySelectorAll('.level-2-skill > h4');
    level2SkillHeaders.forEach(header => {
        header.addEventListener('click', function () {
            // The collapsible content is the .details-container that is the next sibling of the h4
            const content = this.nextElementSibling;
            if (content && content.classList.contains('details-container')) {
                content.classList.toggle('show');
                this.parentElement.classList.toggle('expanded'); // For arrow indicator styling
            }
        });
    });

    // Handle Level 1 skills (h6 headers)
    // These are nested, so we look for h6 within .level-1-skill
    const level1SkillHeaders = document.querySelectorAll('.level-1-skill > h6');
    level1SkillHeaders.forEach(header => {
        header.addEventListener('click', function () {
            // The collapsible content is the .details-container that is the next sibling of the h6
            const content = this.nextElementSibling;
            if (content && content.classList.contains('details-container')) {
                content.classList.toggle('show');
                this.parentElement.classList.toggle('expanded'); // For arrow indicator styling
            }
        });
    });

    // Optional: Expand L2 by default if needed, or specific L1s.
    // For example, to expand the first L2 skill by default:
    // if (level2SkillHeaders.length > 0) {
    //     const firstL2Content = level2SkillHeaders[0].nextElementSibling;
    //     if (firstL2Content && firstL2Content.classList.contains('details-container')) {
    //         firstL2Content.classList.add('show');
    //         level2SkillHeaders[0].parentElement.classList.add('expanded');
    //     }
    // }
}); 